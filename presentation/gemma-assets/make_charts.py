"""Bar charts for slides 6 and 7 of the panel deck, with exact values from the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LIGHT, MID, NAVY, TEXT, GRID = "#74B8E5", "#2689CA", "#083D5E", "#2B3440", "#D9DEE5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT})

def style(ax, ylabel):
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=10)

def label(ax, bars, values, fmt):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, fmt.format(v),
                ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)

# Slide 6: Logistics, valid plans (%), 15 problems x 2 repetitions, 21,000 tokens.
systems = ["Single agent", "Octopus", "Bee", "Direct", "Ant"]
sonnet = [73, 67, 60, 57, 0]
luna = [90, 87, 47, 47, 0]
x = np.arange(len(systems)); w = 0.36
fig, ax = plt.subplots(figsize=(12.9, 7.2), dpi=200)
b1 = ax.bar(x - w / 2, sonnet, w, color=LIGHT, label="Claude Sonnet 4.6")
b2 = ax.bar(x + w / 2, luna, w, color=MID, label="GPT-6 Luna")
label(ax, b1, sonnet, "{}%"); label(ax, b2, luna, "{}%")
ax.set_xticks(x, systems)
style(ax, "Valid plans (%)")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig("slide6-planning-chart.png", transparent=True)

# Slide 7: AIME 2020-2025, 148 held-out problems, GPT-6 Luna, accuracy (%).
names = ["Single agent", "Majority\n(5 answers)", "Quorum bee\n(5 scouts)"]
acc = [70.9, 84.5, 86.5]
fig, ax = plt.subplots(figsize=(10.1, 6.6), dpi=200)
bars = ax.bar(names, acc, 0.55, color=[LIGHT, MID, NAVY])
label(ax, bars, acc, "{}%")
style(ax, "Problems solved (%)")
fig.tight_layout()
fig.savefig("slide7-maths-chart.png", transparent=True)
