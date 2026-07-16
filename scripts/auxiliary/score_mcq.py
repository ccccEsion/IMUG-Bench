import argparse
import json
import re

from common import (
    ensure_directory,
    get_paths,
    load_jsonl_list,
    model_score_path,
    model_score_dir,
    report,
    report_validation,
    select_models,
    validate_model_outputs,
)


def extract_choices(text):
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r"\b([A-G])\b", text))


def format_multiplier(response):
    if not response:
        return 0
    response = response.strip()
    if re.fullmatch(r"[A-G]+", response):
        return 1.0
    if re.fullmatch(r"[A-G\s,.;\-]+", response):
        return 0.75
    return 0.5


def calculate_score(answer, response):
    expected = extract_choices(answer)
    predicted = extract_choices(response)
    if not expected:
        return 0.0
    correct = len(predicted & expected)
    incorrect = len(predicted - expected)
    return round(max(0.0, (correct - incorrect) / len(expected)) * format_multiplier(response), 4)


def load_dynamic_answers(path):
    if not path.exists():
        return {}
    return {
        (item["domain"], item["subdomain"], str(item["sample_id"]), int(item["turn"])): item[
            "determined_answer"
        ]
        for item in load_jsonl_list(path)
    }


def load_model_responses(path):
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return {int(item["turn"]): item.get("model_response", "") for item in result.get("tasks", [])}


def score_model(model, benchmark, paths):
    dynamic_answers = load_dynamic_answers(model_score_path(paths, "dynamic_answers", model, f"{model}_dynamic_answer.jsonl"))
    records = []
    missing_dynamic = 0
    missing_turns = 0
    for item in benchmark:
        domain = item.get("domain")
        subdomain = item.get("subdomain")
        sample_id = str(item.get("sample_id"))
        result_path = paths["model_outputs"] / model / domain / subdomain / f"question_{sample_id}" / "result.json"
        if not result_path.exists():
            continue
        try:
            responses = load_model_responses(result_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            report("mcq_score", "failed", model=model, sample_id=sample_id, error=str(exc))
            continue
        scores = []
        for task in item.get("tasks", []):
            if task.get("modality") != "text":
                continue
            turn = int(task["turn"])
            answer = task.get("output", {}).get("answer", "")
            if not answer:
                continue
            if "<DYNAMIC>" in str(answer):
                answer = dynamic_answers.get((domain, subdomain, sample_id, turn))
                if answer is None:
                    missing_dynamic += 1
                    continue
            if turn not in responses:
                missing_turns += 1
            scores.append({"turn": turn, "score": calculate_score(answer, responses.get(turn, ""))})
        if scores:
            records.append(
                {
                    "domain": domain,
                    "subdomain": subdomain,
                    "sample_id": sample_id,
                    "mcq_scores": scores,
                }
            )
    return records, missing_dynamic, missing_turns


def main():
    parser = argparse.ArgumentParser(description="Score multiple-choice model outputs.")
    parser.add_argument("--models", default="ALL", help="Comma-separated model directories, or ALL.")
    parser.add_argument("--benchmark")
    parser.add_argument("--model-output-dir")
    parser.add_argument("--dynamic-answer-dir")
    parser.add_argument("--mcq-score-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true", help="Stop when model output validation fails.")
    args = parser.parse_args()

    paths = get_paths(args)
    models = select_models(args.models, paths["model_outputs"])
    if not models:
        raise SystemExit("No model output directories found.")
    benchmark = load_jsonl_list(paths["benchmark"])
    ensure_directory(paths["mcq_scores"])
    report("mcq_score", "started", models=models)
    for model in models:
        issues = validate_model_outputs(benchmark, model, paths["model_outputs"], check_text=True)
        report_validation("mcq_score", model, issues)
        if args.strict and issues:
            raise SystemExit(f"Model output validation failed for {model}.")
        records, missing_dynamic, missing_turns = score_model(model, benchmark, paths)
        output_path = model_score_path(paths, "mcq_scores", model, f"{model}_mcq_score.jsonl")
        ensure_directory(model_score_dir(paths, "mcq_scores", model))
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        report(
            "mcq_score",
            "completed",
            model=model,
            records=len(records),
            missing_dynamic=missing_dynamic,
            missing_turns=missing_turns,
        )


if __name__ == "__main__":
    main()
