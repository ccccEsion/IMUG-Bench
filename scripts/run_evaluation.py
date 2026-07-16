import argparse
import getpass
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "auxiliary"))

from common import (
    REPO_ROOT,
    config_path_from_args,
    ensure_directory,
    get_paths,
    load_jsonl_list,
    report,
    report_validation,
    resolve_path,
    select_models,
    validate_model_outputs,
    write_json,
)


def read_existing_config(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prompt_value(label, default="", secret=False):
    suffix = f" [{default}]" if default else ""
    value = getpass.getpass(f"{label}{suffix}: ") if secret else input(f"{label}{suffix}: ")
    return value.strip() or default


def configure(path, current):
    judge = current.get("judge", {})
    stored_paths = current.get("paths", {})
    api_base_url = prompt_value("Judge API base URL", judge.get("api_base_url", ""))
    api_key = prompt_value("Judge API key", judge.get("api_key", ""), secret=True)
    model = prompt_value("Judge model", judge.get("model", ""))
    model_output_dir = prompt_value(
        "Model output directory", stored_paths.get("model_output_dir", "outputs/model_outputs")
    )
    if not all((api_base_url, api_key, model, model_output_dir)):
        raise ValueError("All configuration values are required.")
    config = {
        "judge": {"api_base_url": api_base_url, "api_key": api_key, "model": model},
        "paths": {"model_output_dir": model_output_dir},
    }
    write_json(path, config)
    return config


def build_command(script, config_path, model_output_dir, models, workers=None):
    command = [sys.executable, str(script_path(script)), "--model-output-dir", str(model_output_dir), "--models", ",".join(models)]
    if script in {"resolve_dynamic_answers.py", "score_image.py"}:
        command.extend(["--config", str(config_path)])
    if workers and script in {"resolve_dynamic_answers.py", "score_image.py"}:
        command.extend(["--workers", str(workers)])
    command.append("--strict")
    return command


def script_path(script):
    return SCRIPT_DIR / "auxiliary" / script


def display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generator_for(seed, *parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return random.Random(seed ^ int.from_bytes(digest[:8], "big"))


def create_random_judge_handler(seed):
    class RandomJudgeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                messages = payload.get("messages", [])
                system = next((message.get("content", "") for message in messages if message.get("role") == "system"), "")
                user = next((message.get("content", []) for message in messages if message.get("role") == "user"), [])
                generator = generator_for(seed, json.dumps(payload, sort_keys=True, ensure_ascii=True))
                if "Dynamic Answer Determination Assistant" in system:
                    content = json.dumps(
                        {"determined_answer": generator.choice("ABCDEFG"), "reasoning": "Random smoke-test response."}
                    )
                else:
                    text = "\n".join(part.get("text", "") for part in user if part.get("type") == "text") if isinstance(user, list) else str(user)
                    points = text.split("[Evaluation Points List]:", 1)[-1]
                    point_count = len(re.findall(r"(?m)^\d+\.\s", points))
                    content = json.dumps(
                        {
                            "evaluation_results": [
                                {"point_id": index + 1, "score": generator.choices(range(6), weights=(1, 2, 4, 5, 4, 2), k=1)[0]}
                                for index in range(point_count)
                            ]
                        }
                    )
                response = {
                    "id": "smoke-test-response",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format, *args):
            return

    return RandomJudgeHandler


def run_command(command):
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def is_direct_model_root(path, benchmark):
    domains = {item["domain"] for item in benchmark}
    return any((Path(path) / domain).is_dir() for domain in domains)


def prepare_model_alias(paths, model, benchmark, alias_parent, stage="smoke_test"):
    source = Path(paths["model_outputs"])
    if (source / model).is_dir() or not is_direct_model_root(source, benchmark):
        return
    alias = Path(alias_parent) / model
    ensure_directory(alias.parent)
    if not alias.exists():
        alias.symlink_to(source.resolve(), target_is_directory=True)
    paths["model_outputs"] = alias.parent
    report(stage, "model_root_alias_created", model=model, source=str(source), alias=str(alias))


def run_smoke_test(args, paths, models):
    output_dir = resolve_path(args.smoke_output_dir, Path("outputs/smoke_test"))
    paths["image_scores"] = output_dir
    paths["dynamic_answers"] = output_dir
    paths["mcq_scores"] = output_dir
    proxy_hosts = {"localhost", "127.0.0.1"}
    for variable in ("NO_PROXY", "no_proxy"):
        proxy_hosts.update(host.strip() for host in os.environ.get(variable, "").split(",") if host.strip())
        os.environ[variable] = ",".join(sorted(proxy_hosts))
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(variable, None)
    benchmark = load_jsonl_list(paths["benchmark"])
    if len(models) == 1:
        prepare_model_alias(paths, models[0], benchmark, output_dir / ".model_inputs")
    failed_models = []
    for model in models:
        issues = validate_model_outputs(benchmark, model, paths["model_outputs"], check_text=True, check_images=True)
        report_validation("smoke_test", model, issues)
        if issues:
            failed_models.append(model)
    if failed_models:
        raise SystemExit(f"Model output validation failed for: {', '.join(failed_models)}")
    model_value = ",".join(models)
    server = ThreadingHTTPServer(("127.0.0.1", args.smoke_port), create_random_judge_handler(args.seed))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config_path = output_dir / "random_judge_config.json"
    write_json(
        config_path,
        {
            "judge": {"api_base_url": f"http://127.0.0.1:{port}/v1", "api_key": "smoke-test", "model": "random-judge"},
            "paths": {"model_output_dir": str(paths["model_outputs"])},
        },
    )
    report("smoke_test", "random_judge_server_started", host="127.0.0.1", port=port, seed=args.seed)
    try:
        workflow = (
            ("resolve_dynamic_answers.py", ["--config", str(config_path), "--output-dir", str(output_dir)]),
            ("score_image.py", ["--config", str(config_path), "--output-dir", str(output_dir)]),
            ("score_grid.py", ["--output-dir", str(output_dir)]),
            ("score_mcq.py", ["--output-dir", str(output_dir)]),
        )
        for script, extra_args in workflow:
            report("smoke_test", "stage_started", script=script)
            command = [
                sys.executable,
                str(script_path(script)),
                "--models",
                model_value,
                "--benchmark",
                str(paths["benchmark"]),
                "--model-output-dir",
                str(paths["model_outputs"]),
                *extra_args,
                "--strict",
            ]
            if script in {"resolve_dynamic_answers.py", "score_image.py"}:
                command.extend(["--workers", str(args.workers)])
            run_command(command)
            report("smoke_test", "stage_completed", script=script)
        if not args.skip_summary:
            summary_output_dir = output_dir / models[0] / "summary" if len(models) == 1 else output_dir / "summary"
            run_command(
                [
                    sys.executable,
                    str(script_path("summarize_results.py")),
                    "--models",
                    model_value,
                    "--image-score-dir",
                    str(paths["image_scores"]),
                    "--mcq-score-dir",
                    str(paths["mcq_scores"]),
                    "--output-dir",
                    str(summary_output_dir),
                ]
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        report("smoke_test", "random_judge_server_stopped", port=port)
    report("smoke_test", "completed", models=models, output_dir=str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Run the IMUG scoring workflow.")
    parser.add_argument("--config", help="Path for the local configuration file.")
    parser.add_argument("--models", default="ALL", help="Comma-separated model directories, or ALL.")
    parser.add_argument("--model", dest="models", help="One evaluated model-output directory.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark")
    parser.add_argument("--dynamic-references")
    parser.add_argument("--model-output-dir", help="Directory containing model output directories.")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--reconfigure", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run offline validation and random-judge scoring.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for --smoke-test.")
    parser.add_argument("--smoke-output-dir", default="outputs/smoke_test")
    parser.add_argument("--smoke-port", type=int, default=0, help="Local port for the random judge server; 0 selects a free port.")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        paths = get_paths(args)
        benchmark = load_jsonl_list(paths["benchmark"])
        requested = args.models.split(",") if isinstance(args.models, str) else args.models
        requested = [item.strip() for item in requested if item.strip()] if requested else []
        if len(requested) == 1 and is_direct_model_root(paths["model_outputs"], benchmark):
            models = requested
        else:
            models = select_models(args.models, paths["model_outputs"])
        if not models:
            raise SystemExit("No model output directories found.")
        report("evaluation", "smoke_test_started", models=models)
        run_smoke_test(args, paths, models)
        return

    config_path = config_path_from_args(args)
    current = read_existing_config(config_path)
    if args.reconfigure or not current:
        config = configure(config_path, current)
        report("evaluation", "configured", config=display_path(config_path))
    else:
        config = current
        report("evaluation", "configuration_loaded", config=display_path(config_path))
    if args.setup_only:
        return

    configured_dir = config.get("paths", {}).get("model_output_dir", "outputs/model_outputs")
    model_output_dir = resolve_path(args.model_output_dir, configured_dir)
    paths = get_paths(args)
    paths["model_outputs"] = model_output_dir
    benchmark = load_jsonl_list(paths["benchmark"])
    requested = args.models.split(",") if isinstance(args.models, str) else args.models
    requested = [item.strip() for item in requested if item.strip()] if requested else []
    if len(requested) == 1 and is_direct_model_root(model_output_dir, benchmark):
        models = requested
        prepare_model_alias(paths, models[0], benchmark, OUTPUT_DIR / ".model_inputs", stage="evaluation")
        model_output_dir = paths["model_outputs"]
    else:
        models = select_models(args.models, model_output_dir)
    if not models:
        raise SystemExit("No model output directories found.")
    workflow = [
        "resolve_dynamic_answers.py",
        "score_image.py",
        "score_grid.py",
        "score_mcq.py",
    ]
    report("evaluation", "started", models=models)
    for script in workflow:
        report("evaluation", "stage_started", script=script)
        command = build_command(script, config_path, model_output_dir, models, args.workers)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode:
            report("evaluation", "stage_failed", script=script, returncode=result.returncode)
            raise SystemExit(result.returncode)
        report("evaluation", "stage_completed", script=script)
    report("evaluation", "completed", models=models)


if __name__ == "__main__":
    main()
