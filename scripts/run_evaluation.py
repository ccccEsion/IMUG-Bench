import argparse
import base64
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from auxiliary.common import REPO_ROOT, ensure_directory, report, resolve_path


DEFAULT_API_URL = "http://127.0.0.1:8000/infer"
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 3


def setup_logging(log_path, clean_log):
    log_path = Path(log_path)
    ensure_directory(log_path.parent)
    if clean_log and log_path.exists():
        log_path.unlink()

    logger = logging.getLogger("imug_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def encode_image_file(images_dir, image_name):
    image_path = Path(image_name)
    if not image_path.is_absolute():
        image_path = Path(images_dir) / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def save_base64_image(value, save_path):
    if "," in value:
        value = value.split(",", 1)[1]
    image_bytes = base64.b64decode(value)
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    save_path.write_bytes(image_bytes)


def call_api_balanced(payload, api_urls, retries, timeout, logger):
    for attempt in range(retries):
        target_url = random.choice(api_urls)
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(target_url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Server response must be a JSON object")
            return result
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Request failed at %s: %s", target_url, exc)
            if attempt + 1 < retries:
                time.sleep(1)
    return None


def get_output_paths(data, output_root, model):
    sample_id = str(data.get("sample_id", "unknown"))
    domain = str(data.get("domain", "Unknown"))
    subdomain = str(data.get("subdomain") or data.get("category") or "Unknown")
    target_dir = Path(output_root) / model / domain / subdomain / f"question_{sample_id}"
    return target_dir, target_dir / "result.json", sample_id


def extract_response(api_result, output_mode):
    if not isinstance(api_result, dict):
        raise ValueError("Server response must be a JSON object")
    response = api_result.get("response")
    if not isinstance(response, dict):
        raise ValueError("Server response must contain a response object")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))

    field = "image" if output_mode == "image_only" else "text"
    value = response.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Server response must contain a non-empty {field} field")
    return value


def build_history(data):
    system_prompt = data.get("system_prompt", "")
    if system_prompt and not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")

    history = []
    if system_prompt:
        history.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    return history


def process_single_line(
    line_data,
    model,
    output_root,
    images_dir,
    api_urls,
    retries,
    timeout,
    temperature,
    force,
    logger,
):
    sample_id = "unknown"
    try:
        data = json.loads(line_data)
        if not isinstance(data, dict):
            raise ValueError("Each benchmark line must be a JSON object")

        target_dir, result_path, sample_id = get_output_paths(data, output_root, model)
        if result_path.exists() and not force:
            return sample_id, True, "skipped"
        ensure_directory(target_dir)

        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Benchmark record must contain a non-empty tasks list")

        history = build_history(data)
        for step in tasks:
            if not isinstance(step, dict):
                raise ValueError(f"Task in sample {sample_id} must be an object")
            turn = step.get("turn")
            modality = str(step.get("modality", "")).lower()
            if not isinstance(turn, int):
                raise ValueError(f"Invalid turn in sample {sample_id}")
            if modality not in {"text", "image"}:
                raise ValueError(f"Invalid modality in sample {sample_id}, turn {turn}")

            user_content = []
            input_items = step.get("input")
            if not isinstance(input_items, list) or not input_items:
                raise ValueError(f"Missing input in sample {sample_id}, turn {turn}")

            for item in input_items:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid input item in sample {sample_id}, turn {turn}")
                for key, value in item.items():
                    if key == "text":
                        if not isinstance(value, str):
                            raise ValueError(f"Invalid text input in sample {sample_id}, turn {turn}")
                        user_content.append({"type": "text", "text": value})
                    elif key == "image":
                        user_content.append(
                            {
                                "type": "image",
                                "image": encode_image_file(images_dir, value),
                            }
                        )

            if not user_content:
                raise ValueError(f"Turn {turn} in sample {sample_id} has no usable input")

            history.append({"role": "user", "content": user_content})
            output_mode = "image_only" if modality == "image" else "text_only"
            payload = {
                "history": history,
                "output_mode": output_mode,
                "temperature": temperature,
            }

            assistant_content = []
            try:
                api_result = call_api_balanced(
                    payload, api_urls, retries, timeout, logger
                )
                value = extract_response(api_result, output_mode)
                if output_mode == "text_only":
                    step["model_response"] = value
                    assistant_content.append({"type": "text", "text": value})
                else:
                    output_name = f"turn_{turn}_output.png"
                    save_base64_image(value, target_dir / output_name)
                    step["model_response"] = output_name
                    assistant_content.append({"type": "image", "image": value})
            except Exception as exc:
                logger.warning("Sample %s turn %s failed: %s", sample_id, turn, exc)
                step["model_response"] = f"ERROR: {exc}"
                assistant_content.append({"type": "text", "text": "Error"})

            history.append({"role": "assistant", "content": assistant_content})

        data.pop("system_prompt", None)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        return sample_id, True, "completed"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Sample %s failed: %s", sample_id, exc)
        return sample_id, False, str(exc)


def validate_model_name(model):
    if not model or Path(model).name != model or model in {".", ".."}:
        raise ValueError("--model must be a single directory name")


def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluation against an IMUG server.")
    parser.add_argument("--model", required=True, help="Name used for the model output directory.")
    parser.add_argument(
        "--api-url",
        dest="api_urls",
        action="append",
        default=None,
        help="Model server endpoint. Repeat this option to use multiple workers.",
    )
    parser.add_argument("--benchmark", default="data/benchmark.jsonl")
    parser.add_argument("--images", default="data/images")
    parser.add_argument(
        "--model-output-dir",
        "--output-dir",
        dest="output_dir",
        default="outputs/model_outputs",
        help="Directory containing per-model output directories.",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--start-index", "--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean-log", "--clean_log", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_model_name(args.model)
    if args.start_index < 0 or args.limit < 0:
        raise SystemExit("--start-index and --limit must be non-negative")
    if args.max_retries < 1 or args.timeout < 1:
        raise SystemExit("--max-retries and --timeout must be positive")

    api_urls = args.api_urls or [DEFAULT_API_URL]
    workers = args.workers or len(api_urls) * 2
    if workers < 1:
        raise SystemExit("--workers must be positive")

    benchmark_path = resolve_path(args.benchmark, REPO_ROOT / "data" / "benchmark.jsonl")
    images_dir = resolve_path(args.images, REPO_ROOT / "data" / "images")
    output_root = resolve_path(args.output_dir, REPO_ROOT / "outputs" / "model_outputs")
    log_path = resolve_path(
        args.log_file,
        REPO_ROOT / "outputs" / "evaluation" / f"{args.model}.log",
    )
    logger = setup_logging(log_path, args.clean_log)

    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_path}")
    if not images_dir.exists():
        raise SystemExit(f"Image directory not found: {images_dir}")

    lines = benchmark_path.read_text(encoding="utf-8").splitlines()
    end_index = len(lines) if args.limit == 0 else min(args.start_index + args.limit, len(lines))
    target_lines = lines[args.start_index:end_index]
    if not target_lines:
        raise SystemExit("No benchmark records selected")

    to_process = []
    skipped = 0
    for line in target_lines:
        try:
            data = json.loads(line)
            _, result_path, _ = get_output_paths(data, output_root, args.model)
            if result_path.exists() and not args.force:
                skipped += 1
            else:
                to_process.append(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            to_process.append(line)

    report(
        "evaluation",
        "started",
        model=args.model,
        records=len(target_lines),
        pending=len(to_process),
        skipped=skipped,
        workers=workers,
    )
    if not to_process:
        report("evaluation", "completed", model=args.model, output_dir=str(output_root))
        return

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_line,
                line,
                args.model,
                output_root,
                images_dir,
                api_urls,
                args.max_retries,
                args.timeout,
                args.temperature,
                args.force,
                logger,
            ): line
            for line in to_process
        }
        for future in as_completed(futures):
            sample_id, success, message = future.result()
            if not success:
                failed += 1
                report(
                    "evaluation",
                    "record_failed",
                    model=args.model,
                    sample_id=sample_id,
                    error=message,
                )

    report(
        "evaluation",
        "completed",
        model=args.model,
        output_dir=str(output_root / args.model),
        failed=failed,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
