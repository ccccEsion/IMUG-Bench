import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "local_config.json"


def report(stage, event, **fields):
    print(json.dumps({"stage": stage, "event": event, **fields}, ensure_ascii=True), flush=True)


def resolve_path(value, default):
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else REPO_ROOT / path


def get_paths(args):
    output_root = resolve_path(getattr(args, "output_dir", None), OUTPUT_DIR)
    score_args = {
        "image_scores": getattr(args, "image_score_dir", None),
        "dynamic_answers": getattr(args, "dynamic_answer_dir", None),
        "mcq_scores": getattr(args, "mcq_score_dir", None),
    }
    return {
        "benchmark": resolve_path(getattr(args, "benchmark", None), DATA_DIR / "benchmark.jsonl"),
        "image_references": resolve_path(
            getattr(args, "image_references", None), DATA_DIR / "image_references.jsonl"
        ),
        "dynamic_references": resolve_path(
            getattr(args, "dynamic_references", None), DATA_DIR / "dynamic_references.jsonl"
        ),
        "grid_gt": resolve_path(getattr(args, "grid_gt", None), DATA_DIR / "grid_gt"),
        "images": resolve_path(getattr(args, "images", None), DATA_DIR / "images"),
        "model_outputs": resolve_path(
            getattr(args, "model_output_dir", None), OUTPUT_DIR / "model_outputs"
        ),
        "image_scores": resolve_path(score_args["image_scores"], output_root),
        "dynamic_answers": resolve_path(score_args["dynamic_answers"], output_root),
        "mcq_scores": resolve_path(score_args["mcq_scores"], output_root),
        "score_dirs_explicit": {key: value is not None for key, value in score_args.items()},
    }


def model_score_dir(paths, key, model):
    base = Path(paths[key])
    if paths.get("score_dirs_explicit", {}).get(key):
        return base
    names = {"image_scores": "score_img", "dynamic_answers": "dynamic_answer", "mcq_scores": "score_mcq"}
    return base / model / names[key]


def model_score_path(paths, key, model, filename):
    return model_score_dir(paths, key, model) / filename


def score_files(root, suffix):
    root = Path(root)
    direct = sorted(root.glob(suffix)) if root.exists() else []
    if direct:
        return direct
    return (
        sorted(root.glob(f"*/{suffix}"))
        + sorted(root.glob(f"*/score_img/{suffix}"))
        + sorted(root.glob(f"*/dynamic_answer/{suffix}"))
        + sorted(root.glob(f"*/score_mcq/{suffix}"))
    )


def load_jsonl_map(path):
    records = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[(item["domain"], item["subdomain"], str(item["sample_id"]))] = item
    return records


def load_jsonl_list(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def list_models(model_output_dir):
    root = Path(model_output_dir)
    if not root.exists():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def select_models(value, model_output_dir):
    available = list_models(model_output_dir)
    if not value or value == ["ALL"] or value == "ALL":
        return available
    requested = value.split(",") if isinstance(value, str) else value
    requested = [item.strip() for item in requested if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Model output directories not found: {', '.join(missing)}")
    return requested


def load_judge_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Judge configuration not found: {path}. Run scripts/run_evaluation.py to create it."
        )
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    judge = config.get("judge", {})
    required = ("api_base_url", "api_key", "model")
    missing = [name for name in required if not judge.get(name)]
    if missing:
        raise ValueError(f"Judge configuration is missing: {', '.join(missing)}")
    return judge


def config_path_from_args(args):
    return resolve_path(getattr(args, "config", None), DEFAULT_CONFIG_PATH)


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.chmod(path, 0o600)


def model_question_dir(model_output_dir, model, domain, subdomain, sample_id):
    return Path(model_output_dir) / model / domain / subdomain / f"question_{sample_id}"


def load_model_result(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("result.json must contain a tasks list")
    return payload


def find_model_image(question_dir, turn, result=None):
    question_dir = Path(question_dir)
    direct_path = question_dir / f"turn_{turn}_output.png"
    if direct_path.exists():
        return direct_path
    if result:
        for task in result.get("tasks", []):
            if task.get("turn") == int(turn):
                response = task.get("model_response")
                candidate = question_dir / response if isinstance(response, str) else None
                if candidate and candidate.exists() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    return candidate
    return None


def validate_model_outputs(benchmark, model, model_output_dir, check_text=True, check_images=False):
    issues = []
    for item in benchmark:
        domain = item["domain"]
        subdomain = item["subdomain"]
        sample_id = str(item["sample_id"])
        tasks = item.get("tasks", [])
        needs_result = check_text and any(task.get("modality") == "text" for task in tasks)
        question_dir = model_question_dir(model_output_dir, model, domain, subdomain, sample_id)
        result_path = question_dir / "result.json"
        result = None
        if needs_result:
            if not result_path.exists():
                issues.append({"code": "missing_result", "domain": domain, "subdomain": subdomain, "sample_id": sample_id})
            else:
                try:
                    result = load_model_result(result_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append({"code": "invalid_result", "domain": domain, "subdomain": subdomain, "sample_id": sample_id, "error": str(exc)})
        elif result_path.exists():
            try:
                result = load_model_result(result_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append({"code": "invalid_result", "domain": domain, "subdomain": subdomain, "sample_id": sample_id, "error": str(exc)})
        response_turns = {}
        if result:
            for response in result["tasks"]:
                if not isinstance(response, dict) or not isinstance(response.get("turn"), int):
                    issues.append({"code": "invalid_result_turn", "domain": domain, "subdomain": subdomain, "sample_id": sample_id})
                    continue
                if not isinstance(response.get("model_response"), str):
                    issues.append({"code": "invalid_model_response", "domain": domain, "subdomain": subdomain, "sample_id": sample_id, "turn": response["turn"]})
                    continue
                response_turns[response["turn"]] = response["model_response"]
        for task in tasks:
            turn = int(task["turn"])
            if check_text and task.get("modality") == "text" and turn not in response_turns:
                issues.append({"code": "missing_text_response", "domain": domain, "subdomain": subdomain, "sample_id": sample_id, "turn": turn})
            if check_images and task.get("modality") == "image" and not find_model_image(question_dir, turn, result):
                issues.append({"code": "missing_image_output", "domain": domain, "subdomain": subdomain, "sample_id": sample_id, "turn": turn})
    return issues


def report_validation(stage, model, issues):
    report(stage, "validation_completed", model=model, valid=not issues, issues=len(issues))
    for issue in issues[:20]:
        report(stage, "validation_error", model=model, **issue)
    if len(issues) > 20:
        report(stage, "validation_error_limit", model=model, omitted=len(issues) - 20)
