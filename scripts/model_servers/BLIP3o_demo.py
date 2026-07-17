import asyncio
import base64
import gc
import os
import random
from io import BytesIO
from typing import List, Literal, Optional

import numpy as np
import torch
import uvicorn
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def required_setting(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before starting the server.")
    return value


MODEL_PATH = required_setting("IMUG_MODEL_PATH")
PROCESSOR_PATH = required_setting("IMUG_PROCESSOR_PATH")
DIFFUSION_PATH = os.path.join(MODEL_PATH, "diffusion-decoder")
DEVICE = os.environ.get("IMUG_DEVICE", "cuda:0")

disable_torch_init()
print("Loading model weights.")

processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
tokenizer, multi_model, _ = load_pretrained_model(MODEL_PATH)
pipe = DiffusionPipeline.from_pretrained(
    DIFFUSION_PATH,
    custom_pipeline="pipeline_llava_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multi_model,
    tokenizer=tokenizer,
    safety_checker=None,
)
pipe.vae.to(DEVICE)
pipe.unet.to(DEVICE)

app = FastAPI()
model_lock = asyncio.Lock()


class ContentItem(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[ContentItem]


class InferRequest(BaseModel):
    history: List[Message]
    output_mode: Literal["text_only", "image_only"]
    seed: int = 12345
    temperature: float = 0.8


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def base64_to_pil(value):
    try:
        if "," in value:
            value = value.split(",", 1)[1]
        image_bytes = base64.b64decode(value)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Image decode failed: {exc}") from exc


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_history(history):
    if not history:
        raise ValueError("history must not be empty")
    if history[-1].role != "user":
        raise ValueError("The last history message must be the current user message.")

    for message_index, message in enumerate(history):
        if not message.content:
            raise ValueError(f"history[{message_index}].content must not be empty")
        if message.role == "system" and message_index != 0:
            raise ValueError("system message is only allowed at history[0]")
        for content_index, item in enumerate(message.content):
            if item.type == "text" and item.text is None:
                raise ValueError(
                    f"history[{message_index}].content[{content_index}] "
                    "is text but has no text field"
                )
            if item.type == "image" and not item.image:
                raise ValueError(
                    f"history[{message_index}].content[{content_index}] "
                    "is image but has no image field"
                )


def build_model_inputs(history):
    conv = conv_templates["qwen"].copy()
    qwen_vl_messages = []
    ordered_images = []

    for message in history:
        conv_parts = []
        qwen_content = []
        for item in message.content:
            if item.type == "text":
                conv_parts.append(item.text)
                qwen_content.append({"type": "text", "text": item.text})
            else:
                image = base64_to_pil(item.image)
                conv_parts.append("<image>")
                ordered_images.append(image)
                qwen_content.append({"type": "image", "image": image})

        full_message_text = "\n".join(conv_parts)
        if message.role == "system":
            conv.system = full_message_text
        elif message.role == "user":
            conv.append_message(conv.roles[0], full_message_text)
        else:
            conv.append_message(conv.roles[1], full_message_text)

        qwen_vl_messages.append({"role": message.role, "content": qwen_content})

    return conv, qwen_vl_messages, ordered_images


@app.post("/infer")
async def infer(request: InferRequest):
    try:
        validate_history(request.history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with model_lock:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            set_seed(request.seed)
            conv, qwen_vl_messages, ordered_images = build_model_inputs(request.history)

            if request.output_mode == "text_only":
                text_prompt = processor.apply_chat_template(
                    qwen_vl_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(qwen_vl_messages)
                processor_kwargs = {
                    "text": [text_prompt],
                    "padding": True,
                    "return_tensors": "pt",
                }
                if image_inputs is not None:
                    processor_kwargs["images"] = image_inputs
                if video_inputs is not None:
                    processor_kwargs["videos"] = video_inputs

                inputs = processor(**processor_kwargs).to(DEVICE)
                generation_kwargs = {"max_new_tokens": 1024}
                if request.temperature > 0:
                    generation_kwargs.update(
                        {"do_sample": True, "temperature": request.temperature}
                    )
                else:
                    generation_kwargs["do_sample"] = False

                with torch.inference_mode():
                    generated_ids = multi_model.generate(**inputs, **generation_kwargs)
                generated_only = generated_ids[:, inputs.input_ids.shape[1]:]
                output_text = processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                )[0]
                return {"response": {"text": output_text}}

            conv.append_message(conv.roles[1], None)
            input_data = [conv.get_prompt()] + ordered_images
            with torch.inference_mode():
                output = pipe(input_data, guidance_scale=3.0)

            if hasattr(output, "image") and output.image is not None:
                output_image = output.image
            elif hasattr(output, "images") and output.images and output.images[0] is not None:
                output_image = output.images[0]
            else:
                raise RuntimeError("Pipeline returned no image output")
            if not isinstance(output_image, Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(output_image)}")
            return {"response": {"image": pil_to_base64(output_image)}}
        except HTTPException:
            raise
        except Exception as exc:
            return {"response": {"error": str(exc)}}
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the BLIP3-o model server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
