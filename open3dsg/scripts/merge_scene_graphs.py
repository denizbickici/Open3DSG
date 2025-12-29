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
    # Merge policy: per object/pair, take the most frequent predicted label; tie-break by sort order.
    label_counts = defaultdict(Counter)
    idx_counts = defaultdict(Counter)
    gt_labels = {}

    for graph in graphs:
        for obj in graph.get("objects", []):
            if not isinstance(obj, dict):
                continue
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
            if not isinstance(edge, dict):
                continue
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


def merge_object_files(graphs):
    # Merge object JSONs: majority-vote object_tag, keep first bbox, union frame detections by frame_id (max visibility).
    tag_counts = defaultdict(Counter)
    bbox_extent = {}
    bbox_center = {}
    frame_info = defaultdict(dict)

    for graph in graphs:
        objects = graph.get("objects", {})
        if not isinstance(objects, dict):
            continue
        for obj in objects.values():
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("id")
            if obj_id is None:
                continue
            obj_id = int(obj_id)
            tag = obj.get("object_tag")
            if tag is not None:
                tag_counts[obj_id][str(tag)] += 1
            if obj_id not in bbox_extent and obj.get("bbox_extent") is not None:
                bbox_extent[obj_id] = obj.get("bbox_extent")
            if obj_id not in bbox_center and obj.get("bbox_center") is not None:
                bbox_center[obj_id] = obj.get("bbox_center")

            for det in obj.get("frame_detections", []):
                if not isinstance(det, dict):
                    continue
                frame_id = det.get("frame_id")
                if frame_id is None:
                    continue
                vis = float(det.get("visibility", 0.0))
                existing = frame_info[obj_id].get(frame_id)
                if existing is None:
                    frame_info[obj_id][frame_id] = det
                    continue
                prev_vis = float(existing.get("visibility", 0.0))
                if vis > prev_vis:
                    frame_info[obj_id][frame_id] = det

    merged_objects = {}
    object_ids = sorted(tag_counts.keys())
    for idx, obj_id in enumerate(object_ids):
        tag, _ = pick_most_common(tag_counts[obj_id])
        frames = list(frame_info[obj_id].values())
        frames = sorted(frames, key=lambda x: float(x.get("visibility", 0.0)), reverse=True)
        merged_objects[f"object_{idx + 1}"] = {
            "id": obj_id,
            "object_tag": tag,
            "bbox_extent": bbox_extent.get(obj_id),
            "bbox_center": bbox_center.get(obj_id),
            "detected_frames": [f.get("frame_id") for f in frames],
            "frame_detections": frames,
        }

    return merged_objects


def merge_edge_files(graphs, object_tags, keep_all_edges=False, drop_none=False):
    # Merge edge JSONs: majority-vote predicate per (object_id1, object_id2); optionally keep all variants.
    edge_counts = defaultdict(int)
    pair_pred_counts = defaultdict(Counter)
    tag_counts = defaultdict(Counter)
    for graph in graphs:
        edges = graph.get("edges", {})
        if not isinstance(edges, dict):
            continue
        for edge in edges.values():
            if not isinstance(edge, dict):
                continue
            s_id = edge.get("object_1_id")
            o_id = edge.get("object_2_id")
            pred = edge.get("relationship")
            if s_id is None or o_id is None or pred is None:
                continue
            s_id = int(s_id)
            o_id = int(o_id)
            pred = str(pred)
            if drop_none and pred == "none":
                continue
            edge_counts[(s_id, o_id, pred)] += 1
            pair_pred_counts[(s_id, o_id)][pred] += 1
            s_tag = edge.get("object_1_tag")
            o_tag = edge.get("object_2_tag")
            if s_tag is not None:
                tag_counts[s_id][str(s_tag)] += 1
            if o_tag is not None:
                tag_counts[o_id][str(o_tag)] += 1

    edges_out = {}
    edge_id = 0
    if keep_all_edges:
        items = sorted(edge_counts.items())
        for (s_id, o_id, pred), _ in items:
            s_tag = object_tags.get(s_id)
            o_tag = object_tags.get(o_id)
            if s_tag is None:
                s_tag, _ = pick_most_common(tag_counts[s_id])
            if o_tag is None:
                o_tag, _ = pick_most_common(tag_counts[o_id])
            desc = f"{s_tag} {pred} {o_tag}" if s_tag and o_tag and pred else ""
            edges_out[f"edge_{edge_id}"] = {
                "edge_id": edge_id,
                "edge_description": desc,
                "object_1_id": s_id,
                "object_1_tag": s_tag,
                "object_2_id": o_id,
                "object_2_tag": o_tag,
                "relationship": pred,
            }
            edge_id += 1
    else:
        items = sorted(pair_pred_counts.items())
        for (s_id, o_id), counts in items:
            pred, _ = pick_most_common(counts)
            if pred is None:
                continue
            if drop_none and pred == "none":
                continue
            s_tag = object_tags.get(s_id)
            o_tag = object_tags.get(o_id)
            if s_tag is None:
                s_tag, _ = pick_most_common(tag_counts[s_id])
            if o_tag is None:
                o_tag, _ = pick_most_common(tag_counts[o_id])
            desc = f"{s_tag} {pred} {o_tag}" if s_tag and o_tag and pred else ""
            edges_out[f"edge_{edge_id}"] = {
                "edge_id": edge_id,
                "edge_description": desc,
                "object_1_id": s_id,
                "object_1_tag": s_tag,
                "object_2_id": o_id,
                "object_2_tag": o_tag,
                "relationship": pred,
            }
            edge_id += 1

    return edges_out


def merge_clip_files(graphs, object_tags):
    # Merge clip embeddings: average embeddings per object_id; object_tag by majority vote if needed.
    tag_counts = defaultdict(Counter)
    sums = {}
    counts = defaultdict(int)
    datasets = set()

    for graph in graphs:
        for ds in graph.get("dataset", []):
            datasets.add(ds)
        objects = graph.get("objects", {})
        if not isinstance(objects, dict):
            continue
        for obj in objects.values():
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("id")
            if obj_id is None:
                continue
            obj_id = int(obj_id)
            tag = obj.get("object_tag")
            if tag is not None:
                tag_counts[obj_id][str(tag)] += 1
            emb = obj.get("clip_embedding")
            if not isinstance(emb, list):
                continue
            if obj_id not in sums:
                sums[obj_id] = [float(x) for x in emb]
                counts[obj_id] = 1
            else:
                if len(sums[obj_id]) != len(emb):
                    continue
                sums[obj_id] = [a + float(b) for a, b in zip(sums[obj_id], emb)]
                counts[obj_id] += 1

    merged = {}
    for idx, obj_id in enumerate(sorted(sums.keys())):
        count = max(1, counts[obj_id])
        avg = [v / count for v in sums[obj_id]]
        tag = object_tags.get(obj_id)
        if tag is None:
            tag, _ = pick_most_common(tag_counts[obj_id])
        merged[f"object_{idx + 1}"] = {
            "id": obj_id,
            "object_tag": tag,
            "clip_embedding": avg,
        }

    return {"dataset": sorted(datasets), "objects": merged}


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)

    groups = defaultdict(lambda: {"combined": [], "objects": [], "edges": [], "clip": []})
    for path in iter_json_files(input_dir, args.recursive):
        if output_dir and path.startswith(os.path.join(output_dir, "")):
            continue
        try:
            with open(path, "r") as f:
                graph = json.load(f)
        except Exception:
            continue
        name = os.path.basename(path)
        if name.endswith("_objects.json"):
            graph_type = "objects"
        elif name.endswith("_edges.json"):
            graph_type = "edges"
        elif name.endswith("_clip_embeddings.json"):
            graph_type = "clip"
        else:
            graph_type = "combined"
        scan_id = graph.get("scan_id")
        if not scan_id:
            continue
        groups[base_scan_id(scan_id)][graph_type].append(graph)

    for scan_id, bucket in groups.items():
        if bucket["combined"]:
            merged = merge_graphs(
                bucket["combined"],
                keep_all_edges=args.keep_all_edges,
                drop_none=args.drop_none,
            )
            merged["scan_id"] = scan_id
            out_path = os.path.join(output_dir, f"{scan_id}.json")
            with open(out_path, "w") as f:
                json.dump(merged, f, indent=2)

        merged_objects = {}
        if bucket["objects"]:
            merged_objects = merge_object_files(bucket["objects"])
            objects_out = {"scan_id": scan_id, "objects": merged_objects}
            out_path = os.path.join(output_dir, f"{scan_id}_objects.json")
            with open(out_path, "w") as f:
                json.dump(objects_out, f, indent=2)

        object_tags = {v["id"]: v.get("object_tag") for v in merged_objects.values()}
        if bucket["edges"]:
            merged_edges = merge_edge_files(
                bucket["edges"],
                object_tags,
                keep_all_edges=args.keep_all_edges,
                drop_none=args.drop_none,
            )
            edges_out = {"scan_id": scan_id, "edges": merged_edges}
            out_path = os.path.join(output_dir, f"{scan_id}_edges.json")
            with open(out_path, "w") as f:
                json.dump(edges_out, f, indent=2)

        if bucket["clip"]:
            merged_clip = merge_clip_files(bucket["clip"], object_tags)
            merged_clip["scan_id"] = scan_id
            out_path = os.path.join(output_dir, f"{scan_id}_clip_embeddings.json")
            with open(out_path, "w") as f:
                json.dump(merged_clip, f, indent=2)

    print(f"Merged {len(groups)} scenes into {output_dir}")


if __name__ == "__main__":
    main()
