import argparse
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from common import (
    ensure_directory,
    get_paths,
    load_jsonl_list,
    model_score_path,
    report,
    report_validation,
    select_models,
    validate_model_outputs,
)


GRID_SUBDOMAINS = ["cubes_coloring", "cubes_completion", "cubes_synthesis"]
file_lock = threading.Lock()


class GridCVAnalyzer:
    def __init__(self):
        self.color_ranges = {
            "r": [((0, 70, 40), (10, 255, 255)), ((160, 70, 40), (180, 255, 255))],
            "g": [((35, 70, 40), (95, 255, 255))],
        }

    def analyze(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Image read failed"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        base_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not base_contours:
            return None, "Grid not found"
        grid_x, grid_y, grid_width, grid_height = cv2.boundingRect(np.concatenate(base_contours))
        eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(eroded)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(solid, [contour], -1, 255, -1)
        grid = cv2.resize(
            solid[grid_y : grid_y + grid_height, grid_x : grid_x + grid_width], (300, 300)
        )
        image = cv2.resize(
            image[grid_y : grid_y + grid_height, grid_x : grid_x + grid_width], (300, 300)
        )
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        matrix = []
        for row_index in range(3):
            row = []
            for column_index in range(3):
                if np.mean(grid[row_index * 100 + 10 : (row_index + 1) * 100 - 10,
                                column_index * 100 + 10 : (column_index + 1) * 100 - 10]) < 15:
                    row.append("0")
                    continue
                cell = hsv_image[row_index * 100 + 20 : (row_index + 1) * 100 - 20,
                                 column_index * 100 + 20 : (column_index + 1) * 100 - 20]
                red_mask = cv2.bitwise_or(
                    cv2.inRange(cell, *self.color_ranges["r"][0]),
                    cv2.inRange(cell, *self.color_ranges["r"][1]),
                )
                green_mask = cv2.inRange(cell, *self.color_ranges["g"][0])
                row.append("r" if np.mean(red_mask) > 30 else "g" if np.mean(green_mask) > 30 else "1")
            matrix.append(row)
        return matrix, "OK"


def get_ground_truth(grid_gt_dir, subdomain, sample_id, turn):
    path = grid_gt_dir / f"Geometric_Coloring__{subdomain}__{sample_id}.jsonl"
    if not path.exists():
        return None
    for item in load_jsonl_list(path):
        if item["turn"] == int(turn):
            return item["grid"]
    return None


def score_logic(detected, ground_truth, point_count):
    scores = {index + 1: 0 for index in range(point_count)}
    if detected is None:
        return [(point_id, 0) for point_id in scores]
    scores[1] = 5
    detected_structure = [[1 if value != "0" else 0 for value in row] for row in detected]
    ground_truth_structure = [[1 if value != "0" else 0 for value in row] for row in ground_truth]
    if detected_structure != ground_truth_structure:
        return [(point_id, scores[point_id]) for point_id in scores]
    scores[2] = 5
    if point_count >= 3:
        red_correct = all(
            detected[row][column] == "r"
            for row in range(3)
            for column in range(3)
            if ground_truth[row][column] == "r"
        )
        no_extra_red = all(
            detected[row][column] != "r"
            for row in range(3)
            for column in range(3)
            if ground_truth[row][column] != "r"
        )
        scores[3] = 5 if red_correct and no_extra_red else 0
    if point_count >= 4:
        green_correct = all(
            detected[row][column] == "g"
            for row in range(3)
            for column in range(3)
            if ground_truth[row][column] == "g"
        )
        no_extra_green = all(
            detected[row][column] != "g"
            for row in range(3)
            for column in range(3)
            if ground_truth[row][column] != "g"
        )
        scores[4] = 5 if green_correct and no_extra_green else 0
    return [(point_id, scores[point_id]) for point_id in sorted(scores)]


def save_score(path, record):
    ensure_directory(path.parent)
    with file_lock:
        records = []
        if path.exists():
            for item in load_jsonl_list(path):
                matches = (
                    item["domain"] == record["domain"]
                    and item["subdomain"] == record["subdomain"]
                    and str(item["sample_id"]) == str(record["sample_id"])
                    and item["turn"] == record["turn"]
                )
                if not matches:
                    records.append(item)
        records.append(record)
        with path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Score geometric coloring outputs.")
    parser.add_argument("--models", default="ALL", help="Comma-separated model directories, or ALL.")
    parser.add_argument("--benchmark")
    parser.add_argument("--grid-gt")
    parser.add_argument("--model-output-dir")
    parser.add_argument("--image-score-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true", help="Stop when model output validation fails.")
    args = parser.parse_args()

    paths = get_paths(args)
    benchmark = load_jsonl_list(paths["benchmark"])
    models = select_models(args.models, paths["model_outputs"])
    if not models:
        raise SystemExit("No model output directories found.")
    analyzer = GridCVAnalyzer()
    tasks = [
        (item, task)
        for item in benchmark
        if item["domain"] == "Geometric_Coloring" and item["subdomain"] in GRID_SUBDOMAINS
        for task in item.get("tasks", [])
        if task.get("modality") == "image"
    ]
    grid_items = [
        item
        for item in benchmark
        if item["domain"] == "Geometric_Coloring" and item["subdomain"] in GRID_SUBDOMAINS
    ]
    total = len(models) * len(tasks)
    completed = 0
    skipped = 0
    report("grid_score", "started", models=models, tasks=total)
    for model in models:
        issues = validate_model_outputs(grid_items, model, paths["model_outputs"], check_text=False, check_images=True)
        report_validation("grid_score", model, issues)
        if args.strict and issues:
            raise SystemExit(f"Model output validation failed for {model}.")
        save_path = model_score_path(paths, "image_scores", model, f"{model}_img_score.jsonl")
        for item, task in tasks:
            subdomain = item["subdomain"]
            sample_id = str(item["sample_id"])
            turn = task["turn"]
            image_path = (
                paths["model_outputs"]
                / model
                / "Geometric_Coloring"
                / subdomain
                / f"question_{sample_id}"
                / f"turn_{turn}_output.png"
            )
            ground_truth = get_ground_truth(paths["grid_gt"], subdomain, sample_id, turn)
            if not image_path.exists() or ground_truth is None:
                skipped += 1
                continue
            detected, _ = analyzer.analyze(image_path)
            point_count = len(task["output"]["evaluation_points"])
            scores = score_logic(detected, ground_truth, point_count)
            save_score(
                save_path,
                {
                    "domain": "Geometric_Coloring",
                    "subdomain": subdomain,
                    "sample_id": sample_id,
                    "turn": int(turn),
                    "scores": [{"point_id": point_id, "score": score} for point_id, score in scores],
                },
            )
            completed += 1
            report("grid_score", "progress", completed=completed, skipped=skipped, total=total)
    report("grid_score", "completed", completed=completed, skipped=skipped, total=total)


if __name__ == "__main__":
    main()
