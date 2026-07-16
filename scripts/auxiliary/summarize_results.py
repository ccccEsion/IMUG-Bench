import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import ensure_directory, get_paths, load_jsonl_list, report, resolve_path, score_files


MODALITIES = ("all", "text", "image")


def load_class_map(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        domain: class_name
        for class_name, domains in config.items()
        for domain in domains
    }


def load_records(image_score_dir, mcq_score_dir, class_map):
    records = []
    for path in score_files(image_score_dir, "*_img_score.jsonl"):
        model = path.name.removesuffix("_img_score.jsonl")
        for item in load_jsonl_list(path):
            scores = [float(point["score"]) for point in item.get("scores", []) if "score" in point]
            if scores:
                records.append(
                    {
                        "model": model,
                        "domain": item["domain"],
                        "class": class_map.get(item["domain"], "Unassigned"),
                        "turn": int(item["turn"]),
                        "modality": "image",
                        "score": sum(scores) / len(scores) * 20,
                    }
                )
    for path in score_files(mcq_score_dir, "*_mcq_score.jsonl"):
        model = path.name.removesuffix("_mcq_score.jsonl")
        for item in load_jsonl_list(path):
            for turn_score in item.get("mcq_scores", []):
                records.append(
                    {
                        "model": model,
                        "domain": item["domain"],
                        "class": class_map.get(item["domain"], "Unassigned"),
                        "turn": int(turn_score["turn"]),
                        "modality": "text",
                        "score": float(turn_score["score"]) * 100,
                    }
                )
    return records


def average(rows):
    return round(sum(row["score"] for row in rows) / len(rows), 2) if rows else None


def grouped(records, field, model):
    result = {}
    values = sorted({record[field] for record in records})
    for value in values:
        result[value] = {}
        for modality in MODALITIES:
            rows = [
                record
                for record in records
                if record["model"] == model
                and record[field] == value
                and (modality == "all" or record["modality"] == modality)
            ]
            result[value][modality] = {"mean": average(rows), "count": len(rows)}
    return result


def overall(records, model):
    return {
        modality: {
            "mean": average(
                [record for record in records if record["model"] == model and (modality == "all" or record["modality"] == modality)]
            ),
            "count": len(
                [record for record in records if record["model"] == model and (modality == "all" or record["modality"] == modality)]
            ),
        }
        for modality in MODALITIES
    }


def format_score(entry):
    return "-" if entry["mean"] is None else f"{entry['mean']:.2f}"


def markdown_table(title, values):
    lines = [f"## {title}", "| Group | All | Text | Image |", "| --- | ---: | ---: | ---: |"]
    for name, scores in values.items():
        lines.append(
            f"| {name} | {format_score(scores['all'])} | {format_score(scores['text'])} | {format_score(scores['image'])} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize IMUG scores as percentages.")
    parser.add_argument("--image-score-dir")
    parser.add_argument("--mcq-score-dir")
    parser.add_argument("--class-config", help="JSON mapping from class names to domain lists.")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", default="ALL", help="Comma-separated model names, or ALL.")
    args = parser.parse_args()

    paths = get_paths(args)
    class_config = resolve_path(args.class_config, Path("config/domain_classes.json"))
    class_map = load_class_map(class_config)
    records = load_records(paths["image_scores"], paths["mcq_scores"], class_map)
    available_models = sorted({record["model"] for record in records})
    models = available_models if args.models == "ALL" else [name.strip() for name in args.models.split(",") if name.strip()]
    missing = sorted(set(models) - set(available_models))
    if missing:
        raise SystemExit(f"Score files not found for: {', '.join(missing)}")
    if not records:
        raise SystemExit("No image or MCQ score records found.")

    default_output = Path("outputs") / models[0] / "summary" if len(models) == 1 else Path("outputs/summary")
    output_dir = resolve_path(args.output_dir, default_output)
    ensure_directory(output_dir)
    report("summary", "started", models=models, records=len(records))
    payload = {"score_unit": "percentage", "models": {}}
    reports = []
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        result = {
            "overall": overall(model_records, model),
            "by_domain": grouped(model_records, "domain", model),
            "by_class": grouped(model_records, "class", model),
            "by_turn": grouped(model_records, "turn", model),
        }
        payload["models"][model] = result
        reports.extend(
            [
                f"# {model}",
                markdown_table("Overall", {"Overall": result["overall"]}),
                markdown_table("By Domain", result["by_domain"]),
                markdown_table("By Class", result["by_class"]),
                markdown_table("By Turn", result["by_turn"]),
                "",
            ]
        )
        report("summary", "model_completed", model=model, records=len(model_records))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(reports), encoding="utf-8")
    print("\n".join(reports))
    if not class_map:
        report("summary", "class_mapping_missing", expected=str(class_config))
    report("summary", "completed", output_dir=str(output_dir))


if __name__ == "__main__":
    main()
