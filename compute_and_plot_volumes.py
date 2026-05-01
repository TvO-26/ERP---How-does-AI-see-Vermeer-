"""
compute_and_plot_volumes.py
---------------------------
Reproduces the per-painting object-count comparison between
  - Ground Truth (GT)
  - YOLOv10x predictions
  - Gemini LLM Interpretation

Outputs:
  - prints the painting counts and per-painting count summaries
  - saves fig_volume_per_painting.png next to this script

Run from the ERP folder:
    python compute_and_plot_volumes.py

Required: pandas, matplotlib  (install with: pip install pandas matplotlib)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ERP = Path(__file__).resolve().parent

# ---------------------------------------------------------------- file paths
GT_CSV   = ERP / "yolo" / "Image Annotation - Annotations (updated).csv"
YOLO_CSV = ERP / "yolo" / "yolo_outputs" / "yolo_predictions_yolov10x.csv"
# Gemini / LLM interpretation file. Adjust the path if yours sits elsewhere.
LLM_CSV  = ERP / "Image Annotation - LLM Interpretation.csv"
if not LLM_CSV.exists():
    # fall back to the upload path used during the analysis session
    alt = Path.home() / "Downloads" / "Image Annotation - LLM Interpretation.csv"
    if alt.exists():
        LLM_CSV = alt

OUT_FIG  = ERP / "yolo" / "yolo_outputs" / "figures" / "fig_volume_per_painting.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- load tables
gt   = pd.read_csv(GT_CSV)
yolo = pd.read_csv(YOLO_CSV)
llm  = pd.read_csv(LLM_CSV)

# In the LLM CSV the Picture ID is only filled on the first row of each
# painting; forward-fill it so every object row carries its Picture ID.
pid_col_llm = "Picture ID" if "Picture ID" in llm.columns else "Picture_ID"
llm[pid_col_llm] = llm[pid_col_llm].ffill()

# ---------------------------------------------------- painting-level coverage
n_gt   = gt["Picture ID"].nunique()
n_yolo = yolo["pic_base"].nunique()
n_llm  = llm[pid_col_llm].nunique()

print("Painting coverage")
print(f"  GT:     {n_gt} paintings")
print(f"  YOLO:   {n_yolo} paintings")
print(f"  Gemini: {n_llm} paintings")
print()

# -------------------------------------------------- per-painting object count
# GT has a Count column per object instance; sum it per painting.
gt_counts   = gt.groupby("Picture ID").size()  # rows per painting (matches Gemini/YOLO)
yolo_counts = yolo.groupby("pic_base").size()
llm_counts  = llm.groupby(pid_col_llm).size()

summary = pd.DataFrame({
    "GT":     gt_counts.describe(),
    "YOLO":   yolo_counts.describe(),
    "Gemini": llm_counts.describe(),
}).round(2)
print("Per-painting object-count summary")
print(summary)
print()

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7, 4.5))

data   = [gt_counts.values, llm_counts.values, yolo_counts.values]
labels = [f"GT\n(n={n_gt})", f"Gemini\n(n={n_llm})", f"YOLO\n(n={n_yolo})"]
colors = ["#2b7a78", "#def2f1", "#3aafa9"]

# violin = distribution shape, boxplot overlay = quartiles + median
parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
for body, c in zip(parts["bodies"], colors):
    body.set_facecolor(c)
    body.set_alpha(0.55)
    body.set_edgecolor("black")

bp = ax.boxplot(data, widths=0.18, patch_artist=True,
                medianprops=dict(color="black"))
for box, c in zip(bp["boxes"], colors):
    box.set_facecolor(c)
    box.set_alpha(0.9)

ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels)
ax.set_ylabel("Objects per painting")
ax.set_title("Per-painting object volume by source")
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200)
print(f"Saved figure to: {OUT_FIG}")
