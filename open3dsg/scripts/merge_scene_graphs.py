#!/usr/bin/env python3
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import json
import os
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge subgraph-level scene graph JSONs into per-scene graphs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with subgraph JSONs produced by --dump_scene_graphs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for merged graphs (default: <input_dir>/merged).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for JSON files under input_dir.",
    )
    parser.add_argument(
        "--keep_all_edges",
        action="store_true",
        help="Keep all predicate variants per (subject_id, object_id) pair.",
    )
    parser.add_argument(
        "--drop_none",
        action="store_true",
        help="Drop edges whose predicate is 'none'.",
    )
    return parser.parse_args()


def iter_json_files(input_dir, recursive):
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if name.endswith(".json"):
                    yield os.path.join(root, name)
    else:
        for name in os.listdir(input_dir):
            if name.endswith(".json"):
                yield os.path.join(input_dir, name)


def base_scan_id(scan_id):
    # Subgraphs are stored as <scan_id>-<hex_split>. Remove the split suffix only.
    if "-" not in scan_id:
        return scan_id
    prefix, suffix = scan_id.rsplit("-", 1)
    if len(suffix) == 1 and suffix.lower() in "0123456789abcdef":
        return prefix
    return scan_id


def pick_most_common(counter):
    if not counter:
        return None, 0
    best_count = max(counter.values())
    candidates = [k for k, v in counter.items() if v == best_count]
    candidates = sorted(candidates, key=lambda x: str(x))
    return candidates[0], best_count


def merge_graphs(graphs, keep_all_edges=False, drop_none=False):
    label_counts = defaultdict(Counter)
    idx_counts = defaultdict(Counter)
    gt_labels = {}

    for graph in graphs:
        for obj in graph.get("objects", []):
            obj_id = obj.get("object_id")
            if obj_id is None:
                continue
            obj_id = int(obj_id)
            pred_label = obj.get("pred_label")
            if pred_label is not None:
                label_counts[obj_id][str(pred_label)] += 1
            pred_idx = obj.get("pred_label_idx")
            if pred_idx is not None:
                idx_counts[obj_id][int(pred_idx)] += 1
            gt_label = obj.get("gt_label")
            if gt_label is not None:
                gt_labels[obj_id] = gt_label

    objects = []
    for obj_id in sorted(label_counts.keys()):
        pred_label, label_count = pick_most_common(label_counts[obj_id])
        pred_idx, idx_count = pick_most_common(idx_counts[obj_id])
        node = {
            "index": len(objects),
            "object_id": obj_id,
            "pred_label": pred_label,
            "pred_label_idx": pred_idx,
            "count": label_count,
        }
        if obj_id in gt_labels:
            node["gt_label"] = gt_labels[obj_id]
        objects.append(node)

    edge_counts = defaultdict(int)
    pair_pred_counts = defaultdict(Counter)
    for graph in graphs:
        for edge in graph.get("edges", []):
            s_id = edge.get("subject_id")
            o_id = edge.get("object_id")
            pred = edge.get("predicate")
            if s_id is None or o_id is None or pred is None:
                continue
            s_id = int(s_id)
            o_id = int(o_id)
            pred = str(pred)
            if drop_none and pred == "none":
                continue
            edge_counts[(s_id, o_id, pred)] += 1
            pair_pred_counts[(s_id, o_id)][pred] += 1

    edges = []
    if keep_all_edges:
        for (s_id, o_id, pred), count in sorted(edge_counts.items()):
            edges.append(
                {
                    "subject_id": s_id,
                    "object_id": o_id,
                    "predicate": pred,
                    "count": count,
                }
            )
    else:
        for (s_id, o_id), counts in sorted(pair_pred_counts.items()):
            pred, count = pick_most_common(counts)
            if pred is None:
                continue
            if drop_none and pred == "none":
                continue
            edges.append(
                {
                    "subject_id": s_id,
                    "object_id": o_id,
                    "predicate": pred,
                    "count": count,
                }
            )

    return {
        "objects": objects,
        "edges": edges,
        "subgraph_count": len(graphs),
    }


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)

    groups = defaultdict(list)
    for path in iter_json_files(input_dir, args.recursive):
        try:
            with open(path, "r") as f:
                graph = json.load(f)
        except Exception:
            continue
        scan_id = graph.get("scan_id")
        if not scan_id:
            continue
        groups[base_scan_id(scan_id)].append(graph)

    for scan_id, graphs in groups.items():
        merged = merge_graphs(
            graphs, keep_all_edges=args.keep_all_edges, drop_none=args.drop_none
        )
        merged["scan_id"] = scan_id
        out_path = os.path.join(output_dir, f"{scan_id}.json")
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)

    print(f"Merged {len(groups)} scenes into {output_dir}")


if __name__ == "__main__":
    main()
