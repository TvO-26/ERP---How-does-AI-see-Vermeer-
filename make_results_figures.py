"""
make_results_figures.py
-----------------------
Generates publication-style figures for every results table in the report
plus side-by-side panels for the qualitative case studies (SK-A-23, SK-A-87).

Run from the ERP folder:
    python make_results_figures.py

Outputs (PNG, 200 dpi) all land in yolo/yolo_outputs/figures/:
    fig_table1_volume_per_painting.png       (also rewritten under old name)
    fig_table2_yolo_per_class.png
    fig_table3_gemini_per_category.png
    fig_table4_mcnemar_results.png
    fig_case_SK-A-23.png
    fig_case_SK-A-87.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

ERP = Path(__file__).resolve().parent
OUT = ERP / "yolo" / "yolo_outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette (consistent across figures, colour-blind safe-ish)
C_GT     = "#2b7a78"
C_GEMINI = "#9b5de5"
C_YOLO   = "#f08a3e"
C_OK     = "#3aafa9"
C_BAD    = "#c44545"
C_NEU    = "#888888"


# ============================================================== TABLE 1 fig
def fig_table1_volume():
    gt   = pd.read_csv(ERP / "yolo" / "Image Annotation - Annotations (updated).csv")
    yolo = pd.read_csv(ERP / "yolo" / "yolo_outputs" / "yolo_predictions_yolov10x.csv")
    llm  = pd.read_csv(ERP / "Image Annotation - LLM Interpretation.csv")
    pid  = "Picture ID" if "Picture ID" in llm.columns else "Picture_ID"
    llm[pid] = llm[pid].ffill()

    # Use raw rows-per-painting for every source so the comparison is fair
    # (the GT "Count" column would inflate GT relative to YOLO/Gemini, which
    # have no equivalent count column).
    gt_counts   = gt.groupby("Picture ID").size()
    yolo_counts = yolo.groupby("pic_base").size()
    llm_counts  = llm.groupby(pid).size()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data   = [gt_counts.values, llm_counts.values, yolo_counts.values]
    labels = [f"GT\n(n={len(gt_counts)})",
              f"Gemini\n(n={len(llm_counts)})",
              f"YOLO\n(n={len(yolo_counts)})"]
    cols   = [C_GT, C_GEMINI, C_YOLO]

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, c in zip(parts["bodies"], cols):
        body.set_facecolor(c); body.set_alpha(0.45); body.set_edgecolor("black")
    bp = ax.boxplot(data, widths=0.18, patch_artist=True,
                    medianprops=dict(color="black"))
    for box, c in zip(bp["boxes"], cols):
        box.set_facecolor(c); box.set_alpha(0.95)

    means = [np.mean(d) for d in data]
    for i, m in enumerate(means, 1):
        ax.scatter(i, m, color="white", edgecolor="black", zorder=5, s=42)
        ax.annotate(f"{m:.1f}", (i, m), xytext=(8, -3),
                    textcoords="offset points", fontsize=9)

    ax.set_xticks(range(1, 4)); ax.set_xticklabels(labels)
    ax.set_ylabel("Objects per painting")
    ax.set_title("Table 1 — Per-painting object volume by source")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    for name in ("fig_table1_volume_per_painting.png",
                 "fig_volume_per_painting.png"):
        fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("OK  table 1")


# ============================================================== TABLE 2 fig
def fig_table2_yolo_per_class():
    df = pd.read_csv(ERP / "yolo" / "yolo_outputs" /
                     "yolo_presence_metrics_yolov10x.csv")
    df = df.sort_values("support (paintings w/ GT)", ascending=True).tail(15)

    classes = df["coco_class"].tolist()
    tp = df["TP"].values; fp = df["FP"].values; fn = df["FN"].values
    f1 = df["F1"].values

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    y = np.arange(len(classes))
    ax.barh(y, tp, color=C_OK,  label="TP")
    ax.barh(y, fn, left=tp, color=C_BAD, label="FN (missed)")
    ax.barh(y, fp, left=tp + fn, color=C_NEU, alpha=0.7, label="FP")

    for i, (t, mi, p, score) in enumerate(zip(tp, fn, fp, f1)):
        ax.text(t + mi + p + 0.6, i, f"F1={score:.2f}",
                va="center", fontsize=8.5)

    ax.set_yticks(y); ax.set_yticklabels(classes)
    ax.set_xlabel("Painting count")
    ax.set_title("Table 2 — YOLOv10x per-class outcomes (top 15 by GT support)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_table2_yolo_per_class.png", dpi=200)
    plt.close(fig)
    print("OK  table 2")


# ============================================================== TABLE 3 fig
def fig_table3_gemini_per_category():
    csv = Path("/sessions/kind-ecstatic-mayer/mnt/outputs/llm_vs_gt_category_metrics.csv")
    if not csv.exists():
        csv = ERP / "llm_vs_gt_category_metrics.csv"
    df = pd.read_csv(csv)
    # Keep only the meaningful 14-ish super-categories with non-trivial support
    df = df[df["support_gt"] >= 5].copy()
    df = df.sort_values("support_gt", ascending=True)

    cats   = df["category"].tolist()
    prec   = df["precision"].fillna(0).values
    rec    = df["recall"].fillna(0).values
    f1     = df["F1"].fillna(0).values
    sup    = df["support_gt"].astype(int).values

    y = np.arange(len(cats))
    h = 0.27
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(y - h, prec, h, color=C_GT, label="Precision")
    ax.barh(y,     rec,  h, color=C_GEMINI, label="Recall")
    ax.barh(y + h, f1,   h, color=C_OK, label="F1")

    for i, s in enumerate(sup):
        ax.text(1.02, i, f"n={s}", va="center", fontsize=8.5, color="black")

    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    ax.set_title("Table 3 — Gemini against GT, per super-category")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_table3_gemini_per_category.png", dpi=200)
    plt.close(fig)
    print("OK  table 3")


# ============================================================== TABLE 4 fig
def fig_table4_head_to_head():
    """Side-by-side YOLO vs Gemini accuracy per super-category."""
    csv = Path("/sessions/kind-ecstatic-mayer/mnt/outputs/mcnemar_yolo_vs_gemini.csv")
    if not csv.exists():
        csv = ERP / "mcnemar_yolo_vs_gemini.csv"
    df = pd.read_csv(csv).copy()
    df["delta"] = df["gemini_acc"] - df["yolo_acc"]
    # sort so largest gaps surface at the top
    df = df.sort_values("delta", ascending=True)

    cats   = df["category"].tolist()
    yolo   = df["yolo_acc"].values
    gemini = df["gemini_acc"].values

    y = np.arange(len(cats))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.barh(y - h/2, yolo,   h, color=C_YOLO,   label="YOLOv10x", edgecolor="black")
    ax.barh(y + h/2, gemini, h, color=C_GEMINI, label="Gemini 3.0", edgecolor="black")

    for i, (a, b) in enumerate(zip(yolo, gemini)):
        ax.text(a + 0.01, i - h/2, f"{a:.2f}", va="center", fontsize=8.5)
        ax.text(b + 0.01, i + h/2, f"{b:.2f}", va="center", fontsize=8.5)

    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_xlabel("Accuracy (fraction of 74 paintings classified correctly)")
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Table 4 — Head-to-head accuracy, YOLOv10x vs Gemini  (n=74)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_table4_head_to_head.png", dpi=200)
    plt.close(fig)
    print("OK  table 4 head-to-head")


def fig_table4_mcnemar():
    csv = Path("/sessions/kind-ecstatic-mayer/mnt/outputs/mcnemar_yolo_vs_gemini.csv")
    if not csv.exists():
        csv = ERP / "mcnemar_yolo_vs_gemini.csv"
    df = pd.read_csv(csv).copy()
    df["delta"] = df["gemini_acc"] - df["yolo_acc"]   # +ve means Gemini better
    df = df.sort_values("delta")

    cats  = df["category"].tolist()
    delta = df["delta"].values
    pvals = df["p_value"].values
    sig   = df["sig"].values
    winner = df["winner"].values

    y = np.arange(len(cats))
    colors = [C_YOLO if w == "YOLO" else C_GEMINI if w == "Gemini" else C_NEU
              for w in winner]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.barh(y, delta, color=colors, edgecolor="black")
    ax.axvline(0, color="black", lw=0.8)

    for i, (d, p, s) in enumerate(zip(delta, pvals, sig)):
        offset = 0.012 if d >= 0 else -0.012
        ha = "left" if d >= 0 else "right"
        label = f"p={p:.3f} {s if s != 'ns' else ''}".strip()
        ax.text(d + offset, i, label, va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_xlabel("Gemini accuracy − YOLO accuracy  (positive ⇒ Gemini better)")
    ax.set_title("Table 4 — McNemar's test, YOLOv10x vs Gemini  (n=74)")
    legend = [
        Patch(facecolor=C_GEMINI, edgecolor="black", label="Gemini wins"),
        Patch(facecolor=C_YOLO,   edgecolor="black", label="YOLO wins"),
        Patch(facecolor=C_NEU,    edgecolor="black", label="Not significant"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    xmax = max(abs(delta.min()), abs(delta.max())) + 0.08
    ax.set_xlim(-xmax, xmax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_table4_mcnemar_results.png", dpi=200)
    plt.close(fig)
    print("OK  table 4")


# ============================================================ CASE STUDIES
def case_study_panel(pic_id, gt_terms, gemini_terms, yolo_terms, out_name):
    """
    Build a 2x2 panel: painting + three labelled lists (GT / Gemini / YOLO).
    """
    img_path = ERP / "yolo" / "images" / f"{pic_id}.jpg"
    overlay_path = ERP / "yolo" / "yolo_outputs" / "overlays_yolov10x" / f"{pic_id}.png"

    fig = plt.figure(figsize=(11, 7.4))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], height_ratios=[1, 1])

    # painting
    axp = fig.add_subplot(gs[:, 0])
    if img_path.exists():
        axp.imshow(Image.open(img_path))
    axp.set_title(f"{pic_id} (Rijksmuseum)", fontsize=11)
    axp.axis("off")

    # YOLO overlay
    axo = fig.add_subplot(gs[0, 1])
    if overlay_path.exists():
        axo.imshow(Image.open(overlay_path))
    axo.set_title("YOLOv10x detections (overlay)", fontsize=10)
    axo.axis("off")

    def list_panel(ax, title, items, color):
        ax.set_facecolor("#fafafa")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.02, 0.95, title, fontsize=10, fontweight="bold", color=color,
                va="top", transform=ax.transAxes)
        body = "\n".join(f"• {t}" for t in items[:14])
        if len(items) > 14:
            body += f"\n  …(+{len(items)-14} more)"
        ax.text(0.02, 0.85, body, fontsize=9, va="top",
                transform=ax.transAxes, family="DejaVu Sans")

    ax_gt   = fig.add_subplot(gs[0, 2])
    ax_gem  = fig.add_subplot(gs[1, 1])
    ax_yolo = fig.add_subplot(gs[1, 2])

    list_panel(ax_gt,   "Ground-truth annotation",       gt_terms,     C_GT)
    list_panel(ax_gem,  "Gemini 3.0 Flash interpretation", gemini_terms, C_GEMINI)
    list_panel(ax_yolo, "YOLOv10x predictions",           yolo_terms,   C_YOLO)

    fig.suptitle(f"Qualitative case study — {pic_id}",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=200)
    plt.close(fig)
    print(f"OK  {out_name}")


def fig_case_studies():
    gt   = pd.read_csv(ERP / "yolo" / "Image Annotation - Annotations (updated).csv")
    yolo = pd.read_csv(ERP / "yolo" / "yolo_outputs" / "yolo_predictions_yolov10x.csv")
    llm  = pd.read_csv(ERP / "Image Annotation - LLM Interpretation.csv")
    pid  = "Picture ID" if "Picture ID" in llm.columns else "Picture_ID"
    llm[pid] = llm[pid].ffill()

    def grab(pic):
        gt_terms = (gt.loc[gt["Picture ID"] == pic, "Object"]
                      .dropna().astype(str).tolist())
        ll_terms = (llm.loc[llm[pid].astype(str).str.startswith(pic),
                            "Identified Object"]
                       .dropna().astype(str).tolist())
        yo_terms = (yolo.loc[yolo["pic_base"] == pic, "coco_class"]
                          .dropna().astype(str).tolist())
        # de-duplicate while keeping order
        def uniq(seq):
            seen, out = set(), []
            for s in seq:
                if s not in seen:
                    out.append(s); seen.add(s)
            return out
        return uniq(gt_terms), uniq(ll_terms), uniq(yo_terms)

    for pic in ("SK-A-23", "SK-A-87"):
        gt_t, llm_t, yolo_t = grab(pic)
        case_study_panel(pic, gt_t, llm_t, yolo_t,
                         f"fig_case_{pic}.png")


if __name__ == "__main__":
    fig_table1_volume()
    fig_table2_yolo_per_class()
    fig_table3_gemini_per_category()
    fig_table4_head_to_head()
    fig_case_studies()
    print(f"\nAll figures saved to: {OUT}")
