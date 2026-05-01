"""
Evaluation helpers: compare YOLO detections against Emily's manual ground truth.

Ground truth comes from `Image Annotation - Annotations (updated).csv` at the
*painting level* (object + count), NOT bounding-box level. So we evaluate:

  1. Presence accuracy   — per COCO-overlap class: precision, recall, F1
  2. Count accuracy      — MAE between YOLO instance count and GT count
  3. Confusion matrix    — which GT objects does YOLO mis-label?
  4. Overlay PNGs        — draw YOLO boxes on the painting, list GT beside it

YOLO predictions are expected as a DataFrame with columns:
    pic_base, coco_class, confidence, x1, y1, x2, y2

Ground truth comes from load_and_normalize() in analysis.py.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---- Class mapping ---------------------------------------------------------

def load_class_map(path: str | Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (coco_to_gt, gt_to_coco) from the JSON map."""
    raw = json.loads(Path(path).read_text())
    coco_to_gt = {k: v for k, v in raw["coco_to_gt"].items() if v}
    gt_to_coco: Dict[str, List[str]] = defaultdict(list)
    for coco, gts in coco_to_gt.items():
        for g in gts:
            gt_to_coco[g].append(coco)
    return coco_to_gt, dict(gt_to_coco)


# ---- Aggregate predictions and GT -----------------------------------------

def aggregate_predictions(pred_df: pd.DataFrame,
                          coco_to_gt: Dict[str, List[str]]) -> pd.DataFrame:
    """Roll YOLO per-box detections into per-painting per-coco-class counts."""
    out = (
        pred_df.groupby(["pic_base", "coco_class"])
        .size()
        .reset_index(name="yolo_count")
    )
    out = out[out["coco_class"].isin(coco_to_gt.keys())]
    return out


def aggregate_ground_truth(ann: pd.DataFrame,
                           gt_to_coco: Dict[str, List[str]]) -> pd.DataFrame:
    """Per painting × coco-class → GT count (summing all GT names mapped to that COCO)."""
    rows = []
    for pic, g in ann.groupby("pic_base"):
        coco_counts: Counter = Counter()
        for _, r in g.iterrows():
            gt_name = r["obj_canonical"]
            if gt_name not in gt_to_coco:
                continue
            cnt = int(r["cnt"])
            for coco in gt_to_coco[gt_name]:
                coco_counts[coco] += cnt
        for coco, c in coco_counts.items():
            rows.append({"pic_base": pic, "coco_class": coco, "gt_count": c})
    return pd.DataFrame(rows)


# ---- Metrics ---------------------------------------------------------------

def presence_metrics(pred_long: pd.DataFrame,
                     gt_long: pd.DataFrame,
                     paintings: List[str]) -> pd.DataFrame:
    """For each COCO class: TP/FP/FN/precision/recall/F1 at the painting level."""
    merged = pd.merge(
        pred_long, gt_long,
        on=["pic_base", "coco_class"], how="outer"
    ).fillna(0)

    rows = []
    for coco, g in merged.groupby("coco_class"):
        pics = list(paintings)
        g = (g.set_index("pic_base")[["yolo_count", "gt_count"]]
               .reindex(pics, fill_value=0)
               .reset_index())
        yolo_present = g["yolo_count"] > 0
        gt_present = g["gt_count"] > 0
        tp = int((yolo_present & gt_present).sum())
        fp = int((yolo_present & ~gt_present).sum())
        fn = int((~yolo_present & gt_present).sum())
        tn = int((~yolo_present & ~gt_present).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        rows.append({
            "coco_class": coco,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 3) if not np.isnan(precision) else None,
            "recall": round(recall, 3) if not np.isnan(recall) else None,
            "F1": round(f1, 3),
            "support (paintings w/ GT)": tp + fn,
        })
    return (pd.DataFrame(rows)
            .sort_values("support (paintings w/ GT)", ascending=False)
            .reset_index(drop=True))


def count_metrics(pred_long: pd.DataFrame,
                  gt_long: pd.DataFrame) -> pd.DataFrame:
    """Mean Absolute Error of instance counts per COCO class, per painting."""
    m = pd.merge(pred_long, gt_long, on=["pic_base", "coco_class"], how="outer").fillna(0)
    m["abs_err"] = (m["yolo_count"] - m["gt_count"]).abs()
    summary = (
        m.groupby("coco_class")
        .agg(MAE=("abs_err", "mean"),
             mean_gt=("gt_count", "mean"),
             mean_yolo=("yolo_count", "mean"),
             n_paintings=("pic_base", "nunique"))
        .round(2)
        .sort_values("MAE", ascending=False)
        .reset_index()
    )
    return summary


def confusion_matrix(pred_long: pd.DataFrame,
                     ann: pd.DataFrame,
                     coco_to_gt: Dict[str, List[str]]) -> pd.DataFrame:
    """For each COCO prediction, what GT objects were in that painting?
    Gives a sense of what YOLO's 'tv' predictions actually sit next to in the GT
    (e.g. are they mostly landing in paintings that contain framed paintings?).
    """
    ann_by_pic = ann.groupby("pic_base")["obj_canonical"].apply(list).to_dict()
    rows = []
    for _, r in pred_long.iterrows():
        pic = r["pic_base"]
        coco = r["coco_class"]
        gt_list = ann_by_pic.get(pic, [])
        mapped = set(coco_to_gt.get(coco, []))
        hit = bool(mapped & set(gt_list))
        for g in gt_list:
            rows.append({"coco_class": coco, "gt_object": g,
                         "mapped_hit": g in mapped})
    cm = (pd.DataFrame(rows)
          .groupby(["coco_class", "gt_object"])
          .size().reset_index(name="cooccurrence"))
    return cm.sort_values(["coco_class", "cooccurrence"], ascending=[True, False])


# ---- Summary overview ------------------------------------------------------

def summary_row(presence: pd.DataFrame, counts: pd.DataFrame) -> Dict[str, float]:
    """One-line headline numbers for a model run."""
    p_macro = presence["precision"].astype(float).mean(skipna=True)
    r_macro = presence["recall"].astype(float).mean(skipna=True)
    f_macro = presence["F1"].astype(float).mean(skipna=True)
    mae_macro = counts["MAE"].astype(float).mean(skipna=True)
    return {
        "macro_precision": round(p_macro, 3),
        "macro_recall": round(r_macro, 3),
        "macro_F1": round(f_macro, 3),
        "mean_count_MAE": round(mae_macro, 3),
    }


# ---- Overlay drawing -------------------------------------------------------

def draw_overlay(image_path: str | Path,
                 pred_df_for_pic: pd.DataFrame,
                 gt_list_for_pic: List[str],
                 out_path: str | Path) -> None:
    """Render the painting with YOLO boxes + GT list on the side."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # side panel width
    panel_w = max(260, W // 3)
    canvas = Image.new("RGB", (W + panel_w, H), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        fb = font

    # cycle colors per class
    palette = ["#e53e3e", "#3182ce", "#38a169", "#d69e2e", "#805ad5",
               "#dd6b20", "#319795", "#b83280", "#2c5282", "#c05621"]
    class_color = {}

    for _, r in pred_df_for_pic.iterrows():
        cls = r["coco_class"]
        color = class_color.setdefault(cls, palette[len(class_color) % len(palette)])
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls} {r['confidence']:.2f}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 2), label, fill="white", font=font)

    # side panel
    draw.text((W + 14, 14), "Ground truth (manual)", fill="#1a202c", font=fb)
    y = 40
    for g, c in Counter(gt_list_for_pic).most_common():
        draw.text((W + 18, y), f"• {g} ×{c}", fill="#2d3748", font=font)
        y += 20
        if y > H - 20:
            draw.text((W + 18, y), "…", fill="#718096", font=font)
            break

    canvas.save(out_path)
