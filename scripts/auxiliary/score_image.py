import argparse
import base64
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from common import (
    config_path_from_args,
    ensure_directory,
    get_paths,
    load_judge_config,
    load_jsonl_list,
    model_score_path,
    report,
    report_validation,
    resolve_path,
    select_models,
    validate_model_outputs,
)


write_lock = threading.Lock()


def encode_image(path):
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def find_image_path(task, model, paths, domain, subdomain, sample_id, turn, image_type):
    if image_type == "input":
        for image in task.get("input", []):
            if "image" in image:
                candidate = paths["images"] / image["image"]
                if candidate.exists():
                    return candidate
        return None

    question_dir = (
        paths["model_outputs"]
        / model
        / domain
        / subdomain
        / f"question_{sample_id}"
    )
    candidate = question_dir / f"turn_{turn}_output.png"
    if candidate.exists():
        return candidate

    result_path = question_dir / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for task_result in result.get("tasks", []):
                if task_result["turn"] == int(turn):
                    image_name = task_result.get("model_response")
                    candidate = question_dir / image_name
                    if isinstance(image_name, str) and candidate.exists():
                        return candidate
        except (KeyError, JSONDecodeError, OSError, TypeError):
            pass
    return None


def load_existing_scores(path):
    if not path.exists():
        return set()
    return {
        (item["domain"], item["subdomain"], str(item["sample_id"]), item["turn"])
        for item in load_jsonl_list(path)
    }


def parse_scores(text):
    json_string = re.search(r"(\{.*\})", text, re.DOTALL).group(1)
    payload = json.loads(json_string)
    return [
        {"point_id": result.get("point_id"), "score": result.get("score")}
        for result in payload.get("evaluation_results", [])
    ]


def evaluate_task(unit, client, judge, prompt_template, paths):
    item, task, references, model = unit
    domain = item["domain"]
    subdomain = item["subdomain"]
    sample_id = str(item["sample_id"])
    turn = task["turn"]
    images = []

    for index, reference in enumerate(references):
        reference_turn = reference["ref_turn"]
        reference_task = next(entry for entry in item["tasks"] if entry["turn"] == int(reference_turn))
        reference_path = find_image_path(
            reference_task,
            model,
            paths,
            domain,
            subdomain,
            sample_id,
            reference_turn,
            reference["ref_type"],
        )
        if reference_path:
            images.append((reference_path, f"Image_{index + 1} (Turn {reference_turn} {reference['ref_type']} image)"))

    output_path = find_image_path(
        task, model, paths, domain, subdomain, sample_id, turn, "output"
    )
    images.append((output_path, f"Image_{len(images) + 1} (Target Image)"))

    points = task["output"].get("evaluation_points", [])
    point_text = "\n".join(f"{index + 1}. {point}" for index, point in enumerate(points))
    image_text = "Sequence description:\n" + "".join(
        f"- {description}\n" for path, description in images if path
    )
    prompt = (
        f"{image_text}\n[Task Instruction Recap]:\n{task['input'][0]['text']}\n\n"
        f"[Evaluation Points List]:\n{point_text}\n\nPlease output JSON results."
    )
    content = [
        {"type": "image_url", "image_url": {"url": encode_image(path)}}
        for path, _ in images
        if path
    ]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=judge["model"],
        messages=[{"role": "system", "content": prompt_template}, {"role": "user", "content": content}],
        temperature=0.1,
    )
    text = response.choices[0].message.content or ""
    return {
        "domain": domain,
        "subdomain": subdomain,
        "sample_id": sample_id,
        "turn": turn,
        "scores": parse_scores(text),
    }


def build_units(benchmark, references, models, paths, existing, skip_existing):
    units = []
    for item in benchmark:
        domain = item["domain"]
        subdomain = item["subdomain"]
        sample_id = str(item["sample_id"])
        for task in item.get("tasks", []):
            if task.get("modality") != "image":
                continue
            turn = task["turn"]
            reference_item = references.get((domain, subdomain, sample_id), {})
            reference_links = next(
                (entry["refs"] for entry in reference_item.get("reference_links", []) if entry["turn_idx"] == int(turn)),
                [],
            )
            for model in models:
                key = (domain, subdomain, sample_id, turn)
                if not (skip_existing and key in existing[model]):
                    units.append((item, task, reference_links, model))
    return units


def append_result(path, result):
    ensure_directory(path.parent)
    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Score image outputs with a configured judge model.")
    parser.add_argument("--config", help="Path to the local judge configuration file.")
    parser.add_argument("--models", default="ALL", help="Comma-separated model directories, or ALL.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark")
    parser.add_argument("--image-references")
    parser.add_argument("--images")
    parser.add_argument("--model-output-dir")
    parser.add_argument("--image-score-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt", default=str(Path(__file__).with_name("eval_prompt.txt")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop when model output validation fails.")
    args = parser.parse_args()

    paths = get_paths(args)
    config_path = config_path_from_args(args)
    judge = load_judge_config(config_path)
    prompt_path = resolve_path(args.prompt, Path(__file__).with_name("eval_prompt.txt"))
    prompt_template = prompt_path.read_text(encoding="utf-8")
    benchmark = load_jsonl_list(paths["benchmark"])
    references = {}
    for entry in load_jsonl_list(paths["image_references"]):
        references[(entry["domain"], entry["subdomain"], str(entry["sample_id"]))] = entry
    models = select_models(args.models, paths["model_outputs"])
    if not models:
        raise SystemExit("No model output directories found.")
    for model in models:
        issues = validate_model_outputs(benchmark, model, paths["model_outputs"], check_text=False, check_images=True)
        report_validation("image_score", model, issues)
        if args.strict and issues:
            raise SystemExit(f"Model output validation failed for {model}.")
    existing = {
        model: load_existing_scores(model_score_path(paths, "image_scores", model, f"{model}_img_score.jsonl"))
        for model in models
    }
    units = build_units(benchmark, references, models, paths, existing, not args.overwrite)
    report("image_score", "started", models=models, tasks=len(units))
    if not units:
        report("image_score", "completed", completed=0, failed=0)
        return

    clients = {}
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for unit in units:
            model = unit[3]
            clients.setdefault(model, OpenAI(base_url=judge["api_base_url"], api_key=judge["api_key"]))
            future = executor.submit(evaluate_task, unit, clients[model], judge, prompt_template, paths)
            futures[future] = unit
        for future in as_completed(futures):
            unit = futures[future]
            model = unit[3]
            try:
                append_result(model_score_path(paths, "image_scores", model, f"{model}_img_score.jsonl"), future.result())
                completed += 1
                report("image_score", "progress", completed=completed, failed=failed, total=len(units))
            except Exception as exc:
                failed += 1
                report(
                    "image_score",
                    "failed",
                    model=model,
                    domain=unit[0]["domain"],
                    subdomain=unit[0]["subdomain"],
                    sample_id=str(unit[0]["sample_id"]),
                    turn=unit[1]["turn"],
                    error=str(exc),
                )
    report("image_score", "completed", completed=completed, failed=failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
