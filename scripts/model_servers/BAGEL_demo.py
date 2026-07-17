import asyncio
import base64
import copy
import gc
import os
from io import BytesIO
from typing import Any, List, Literal, Optional

import torch
import uvicorn
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from fastapi import FastAPI, HTTPException
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from PIL import Image
from pydantic import BaseModel
from modeling.qwen2 import Qwen2Tokenizer


def required_setting(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before starting the server.")
    return value


MODEL_PATH = required_setting("IMUG_MODEL_PATH")
OFFLOAD_FOLDER = os.environ.get("IMUG_OFFLOAD_DIR", "offload")

llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act="gelu_pytorch_tanh",
    latent_patch_size=2,
    max_latent_size=64,
)

print("Loading model weights.")
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)

model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)
device_map = infer_auto_device_map(
    model,
    max_memory={index: "80GiB" for index in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]
first_device = device_map.get(same_device_modules[0], "cuda:0")
for module_name in same_device_modules:
    if module_name in device_map:
        device_map[module_name] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
    device_map=device_map,
    dtype=torch.bfloat16,
    offload_buffers=True,
    offload_folder=OFFLOAD_FOLDER,
    force_hooks=True,
).eval()

BASE_INFERENCE_HYPER = {
    "max_think_token_n": 512,
    "do_sample": True,
    "text_temperature": 0.7,
    "num_timesteps": 50,
    "cfg_text_scale": 3.0,
    "cfg_img_scale": 1.5,
    "cfg_interval": [0.4, 1.0],
    "timestep_shift": 3.0,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
}

app = FastAPI()
model_lock = asyncio.Lock()


class ContentItem(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None


class HistoryItem(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[ContentItem]


class InferRequest(BaseModel):
    history: List[HistoryItem]
    output_mode: Literal["text_only", "image_only"] = "image_only"
    temperature: Optional[float] = 0.7


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def base64_to_image(value):
    try:
        if "," in value:
            value = value.split(",", 1)[1]
        image_bytes = base64.b64decode(value)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Image decode failed: {exc}") from exc


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


def history_to_bagel_input(history: List[HistoryItem]) -> List[Any]:
    input_list: List[Any] = []
    for message in history:
        text_buffer = f"<|im_start|>{message.role}\n"
        for content_item in message.content:
            if content_item.type == "text":
                if content_item.text is None:
                    raise ValueError("Text content item is missing the 'text' field.")
                text_buffer += content_item.text
            else:
                if not content_item.image:
                    raise ValueError("Image content item is missing the 'image' field.")
                if text_buffer:
                    input_list.append(text_buffer)
                    text_buffer = ""
                input_list.append(pil_img2rgb(base64_to_image(content_item.image)))
        text_buffer += "<|im_end|>\n"
        input_list.append(text_buffer)
    input_list.append("<|im_start|>assistant\n")
    return input_list


@app.post("/infer")
async def infer_multimodal(request: InferRequest):
    try:
        validate_history(request.history)
        input_list = history_to_bagel_input(request.history)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input processing failed: {exc}") from exc

    current_hyper = copy.deepcopy(BASE_INFERENCE_HYPER)
    current_hyper["text_temperature"] = request.temperature

    async with model_lock:
        gc.collect()
        torch.cuda.empty_cache()
        temp_inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        original_visual_gen = model.config.visual_gen
        original_visual_und = model.config.visual_und

        try:
            if request.output_mode == "text_only":
                model.config.visual_gen = False
                model.config.visual_und = True
                current_hyper["understanding_output"] = True
            else:
                model.config.visual_gen = True
                model.config.visual_und = False
                current_hyper["understanding_output"] = False

            output_data = await asyncio.to_thread(
                temp_inferencer.interleave_inference,
                input_lists=input_list,
                **current_hyper,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        finally:
            model.config.visual_gen = original_visual_gen
            model.config.visual_und = original_visual_und
            del temp_inferencer
            torch.cuda.empty_cache()

    try:
        raw_output = output_data[0] if isinstance(output_data, list) and output_data else None
        if request.output_mode == "text_only":
            if not isinstance(raw_output, str):
                raise TypeError(f"Expected text output, got {type(raw_output)}")
            response = {"text": raw_output}
        else:
            if not isinstance(raw_output, Image.Image):
                raise TypeError(f"Expected image output, got {type(raw_output)}")
            buffer = BytesIO()
            raw_output.save(buffer, format="PNG")
            response = {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
        return {"response": response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Result encoding failed: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the BAGEL model server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
