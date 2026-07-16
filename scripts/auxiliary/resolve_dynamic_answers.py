import argparse
import base64
import json
import random
import re
import threading
import time
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


def load_dynamic_references(path):
    return {
        (item["domain"], item["subdomain"], str(item["sample_id"]), item["turn_idx"]): item.get(
            "ref_turn_idxs", []
        )
        for item in load_jsonl_list(path)
    }


def encode_image(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def find_output_image(model, paths, domain, subdomain, sample_id, turn):
    path = (
        paths["model_outputs"]
        / model
        / domain
        / subdomain
        / f"question_{sample_id}"
        / f"turn_{turn}_output.png"
    )
    return path if path.exists() else None


def resolve_answer(unit, benchmark, prompt, client, judge, paths):
    model, domain, subdomain, sample_id, turn, reference_turns, raw_answer = unit
    item = benchmark[(domain, subdomain, sample_id)]
    content = []
    reference_text = "### REFERENCE CONTEXT PROVIDED ###\n"
    image_index = 1
    for reference_turn in sorted(reference_turns):
        task = next((entry for entry in item["tasks"] if entry["turn"] == reference_turn), None)
        if not task:
            continue
        question = task["input"][0].get("text", "[Visual Input]")
        image_path = find_output_image(model, paths, domain, subdomain, sample_id, reference_turn)
        if image_path:
            answer = f"[IMAGE_RESPONSE: See Image_{image_index}]"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                }
            )
            image_index += 1
        else:
            answer = task["output"].get("answer", "[Text Response]")
        reference_text += f"Turn {reference_turn}: Q: {question} | AI Resp: {answer}\n"
    target_task = next(entry for entry in item["tasks"] if entry["turn"] == turn)
    choice_type = "MULTIPLE-CHOICE" if domain == "Botany" and subdomain == "natural_growth" else "SINGLE-CHOICE"
    final_prompt = (
        f"{reference_text}\n### TARGET QUESTION (Turn {turn}) ###\n"
        f"Question: {target_task['input'][0]['text']}\n"
        f"Type: {choice_type}\n"
        "Identify the correct options based on the images provided."
    )
    content.insert(0, {"type": "text", "text": final_prompt})
    backoff = 2.0
    while True:
        try:
            response = client.chat.completions.create(
                model=judge["model"],
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                temperature=0.1,
                timeout=150,
            )
            return raw_answer, response.choices[0].message.content or ""
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc):
                time.sleep(backoff + random.random())
                backoff = min(backoff * 1.5, 60.0)
                continue
            raise RuntimeError(str(exc)) from exc


def existing_answers(path):
    if not path.exists():
        return set()
    return {
        (item["domain"], item["subdomain"], str(item["sample_id"]), item["turn"])
        for item in load_jsonl_list(path)
    }


def save_answer(path, unit, raw_answer, response):
    model, domain, subdomain, sample_id, turn, _, _ = unit
    json_string = re.search(r"(\{.*\})", response, re.DOTALL).group(1)
    determined_answer = str(json.loads(json_string).get("determined_answer", "")).upper()
    fixed_options = set(re.findall(r"[A-G]", raw_answer.split("<DYNAMIC>")[0]))
    dynamic_options = set(re.findall(r"[A-G]", determined_answer))
    record = {
        "domain": domain,
        "subdomain": subdomain,
        "sample_id": sample_id,
        "turn": turn,
        "determined_answer": "".join(sorted(fixed_options | dynamic_options)),
    }
    ensure_directory(path.parent)
    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Resolve dynamic benchmark answers with a configured judge model.")
    parser.add_argument("--config", help="Path to the local judge configuration file.")
    parser.add_argument("--models", default="ALL", help="Comma-separated model directories, or ALL.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark")
    parser.add_argument("--dynamic-references")
    parser.add_argument("--model-output-dir")
    parser.add_argument("--dynamic-answer-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt", default=str(Path(__file__).with_name("dynamic_prompt.txt")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop when model output validation fails.")
    args = parser.parse_args()

    paths = get_paths(args)
    judge = load_judge_config(config_path_from_args(args))
    prompt_path = resolve_path(args.prompt, Path(__file__).with_name("dynamic_prompt.txt"))
    prompt = prompt_path.read_text(encoding="utf-8")
    benchmark_items = load_jsonl_list(paths["benchmark"])
    benchmark = {
        (item["domain"], item["subdomain"], str(item["sample_id"])): item for item in benchmark_items
    }
    references = load_dynamic_references(paths["dynamic_references"])
    models = select_models(args.models, paths["model_outputs"])
    if not models:
        raise SystemExit("No model output directories found.")
    for model in models:
        issues = validate_model_outputs(benchmark_items, model, paths["model_outputs"], check_text=False, check_images=True)
        report_validation("dynamic_answer", model, issues)
        if args.strict and issues:
            raise SystemExit(f"Model output validation failed for {model}.")
    units = []
    for model in models:
        output_path = model_score_path(paths, "dynamic_answers", model, f"{model}_dynamic_answer.jsonl")
        completed = set() if args.overwrite else existing_answers(output_path)
        for (domain, subdomain, sample_id, turn), reference_turns in references.items():
            if (domain, subdomain, sample_id, turn) in completed:
                continue
            item = benchmark.get((domain, subdomain, sample_id))
            if item is None:
                continue
            task = next((entry for entry in item["tasks"] if entry["turn"] == turn), None)
            if task is None:
                continue
            units.append(
                (
                    model,
                    domain,
                    subdomain,
                    sample_id,
                    turn,
                    reference_turns,
                    task["output"].get("answer", ""),
                )
            )
    report("dynamic_answer", "started", models=models, tasks=len(units))
    clients = {
        model: OpenAI(base_url=judge["api_base_url"], api_key=judge["api_key"]) for model in models
    }
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(resolve_answer, unit, benchmark, prompt, clients[unit[0]], judge, paths): unit
            for unit in units
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                raw_answer, response = future.result()
                save_answer(model_score_path(paths, "dynamic_answers", unit[0], f"{unit[0]}_dynamic_answer.jsonl"), unit, raw_answer, response)
                completed += 1
                report("dynamic_answer", "progress", completed=completed, failed=failed, total=len(units))
            except Exception as exc:
                failed += 1
                report(
                    "dynamic_answer",
                    "failed",
                    model=unit[0],
                    domain=unit[1],
                    subdomain=unit[2],
                    sample_id=unit[3],
                    turn=unit[4],
                    error=str(exc),
                )
    report("dynamic_answer", "completed", completed=completed, failed=failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
