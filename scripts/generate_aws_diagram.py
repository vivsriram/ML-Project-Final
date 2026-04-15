"""
AWS Architecture — vertical layout, Word-doc width (~6 in wide)
Generates: midpoint_visual_05_aws_architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

C_S3    = "#2E7D32"   # green
C_EC2   = "#6A1B9A"   # purple
C_USER  = "#BF360C"   # burnt orange (dashed — outside AWS)
C_LINE  = "#AAAAAA"

fig, ax = plt.subplots(figsize=(6.2, 8.0))
ax.set_xlim(0, 6.2)
ax.set_ylim(0, 8.0)
ax.axis("off")
fig.patch.set_facecolor("white")

BW = 4.4    # box width
BH = 0.88   # box height
CX = 3.1    # center x

# ── box centres (top → bottom) ────────────────────────────────────────────────
Y = [7.0, 5.4, 3.8, 2.2]   # Local, S3, EC2, TSA Planner

def draw_box(x, y, title, sub, color, dashed=False):
    ls = (0, (5, 4)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch(
        (x - BW/2, y - BH/2), BW, BH,
        boxstyle="round,pad=0.07",
        linewidth=2.0, edgecolor=color,
        linestyle=ls,
        facecolor=color + "18"
    ))
    ax.text(x, y + 0.17, title,
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x, y - 0.22, sub,
            ha="center", va="center",
            fontsize=8.5, color="#444444")

def vert_arrow(y_top, y_bot, label):
    ay_start = y_top - BH/2 - 0.04
    ay_end   = y_bot + BH/2 + 0.04
    ax.annotate("", xy=(CX, ay_end), xytext=(CX, ay_start),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.5, mutation_scale=14))
    my = (ay_start + ay_end) / 2
    ax.text(CX + 0.16, my, label,
            ha="left", va="center",
            fontsize=8.0, color="#666666", style="italic")

# ── Boxes ──────────────────────────────────────────────────────────────────────
draw_box(CX, Y[0], "Local Machine",
         "Train LightGBM · save lgbmModel.pkl",
         C_USER, dashed=True)

draw_box(CX, Y[1], "Amazon S3",
         "lgbmModel.pkl  ·  modelDf.csv (3.2M rows)",
         C_S3)

draw_box(CX, Y[2], "Amazon EC2  (t3.medium)",
         "Streamlit app · live inference · port 8501",
         C_EC2)

draw_box(CX, Y[3], "TSA Planner  (browser)",
         "Select airport, month, day · view forecast",
         C_USER, dashed=True)

# ── Arrows ─────────────────────────────────────────────────────────────────────
vert_arrow(Y[0], Y[1], "upload artifact  (one-time)")
vert_arrow(Y[1], Y[2], "load model at startup")

# Bidirectional arrow between EC2 and TSA Planner
ay_s = Y[2] - BH/2 - 0.04
ay_e = Y[3] + BH/2 + 0.04
ax.annotate("", xy=(CX, ay_e), xytext=(CX, ay_s),
            arrowprops=dict(arrowstyle="<|-|>", color="#555555",
                            lw=1.5, mutation_scale=14))
my = (ay_s + ay_e) / 2
ax.text(CX + 0.16, my, "inputs  /  predictions",
        ha="left", va="center",
        fontsize=8.0, color="#666666", style="italic")

# ── Section bracket labels ─────────────────────────────────────────────────────
# "Training" bracket covers boxes 0-1
bracket_x = 0.28
for (y_top, y_bot, lbl) in [
    (Y[0] + BH/2, Y[1] - BH/2, "TRAINING"),
    (Y[2] + BH/2, Y[3] - BH/2, "SERVING"),
]:
    ax.plot([bracket_x, bracket_x], [y_bot, y_top],
            color="#CCCCCC", lw=2.0, solid_capstyle="round")
    ax.text(bracket_x - 0.08, (y_top + y_bot) / 2, lbl,
            ha="center", va="center", fontsize=7.0,
            fontweight="bold", color="#BBBBBB", rotation=90)

# Dashed divider between training and serving
div_y = (Y[1] - BH/2 + Y[2] + BH/2) / 2
ax.plot([0.5, 5.7], [div_y, div_y],
        color="#EEEEEE", lw=1.2, linestyle="--")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_S3  + "18", edgecolor=C_S3,
                   linewidth=1.8, label="Amazon S3"),
    mpatches.Patch(facecolor=C_EC2 + "18", edgecolor=C_EC2,
                   linewidth=1.8, label="Amazon EC2 (Streamlit)"),
    mpatches.Patch(facecolor=C_USER+ "18", edgecolor=C_USER,
                   linewidth=1.8, linestyle=(0,(5,4)), label="Outside AWS"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, 0.01),
          fontsize=8.5, ncol=3,
          frameon=True, edgecolor="#DDDDDD", facecolor="white")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(CX, 7.78, "Figure A1 — AWS Deployment Architecture",
        ha="center", fontsize=11.5, fontweight="bold", color="#111111")
ax.text(CX, 7.58, "TSA Checkpoint Throughput Forecasting  ·  S3 + EC2 + Streamlit",
        ha="center", fontsize=8.5, color="#888888")

plt.tight_layout(pad=0.2)
plt.savefig("midpoint_visual_05_aws_architecture.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved.")
