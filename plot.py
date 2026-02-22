import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

traj = pd.read_csv("trajectory.csv")
bench = pd.read_csv("results.csv")

fn_name = traj["function"].iloc[0]
optimizers = traj["optimizer"].unique()

functions = {
    "quadratic":   lambda x, y: x**2 + y**2,
    "rosenbrock":  lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
    "himmelblau":  lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
    "beale":       lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
}

colors = {
    "vanilla gradient descent":      "#e6550d",
    "gradient descent + momentum":   "#756bb1",
    "nesterov momentum":             "#31a354",
}

fn = functions[fn_name]
paths = {opt: traj[traj["optimizer"] == opt][["x", "y"]].values for opt in optimizers}

all_x = traj["x"].values
all_y = traj["y"].values
pad = 0.5
xmin, xmax = all_x.min() - pad, all_x.max() + pad
ymin, ymax = all_y.min() - pad, all_y.max() + pad

gx = np.linspace(xmin, xmax, 300)
gy = np.linspace(ymin, ymax, 300)
GX, GY = np.meshgrid(gx, gy)
GZ = fn(GX, GY)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor("white")
fig.suptitle(f"Gradient Descent Trajectories  —  {fn_name}", fontsize=14, fontweight="bold")

lines = {}
dots  = {}
counter_texts = {}

for ax, opt in zip(axes.flat[:3], optimizers):
    ax.set_facecolor("#f7f7f7")
    ax.contourf(GX, GY, np.log1p(GZ), levels=40, cmap="Blues", alpha=0.75)
    ax.contour( GX, GY, np.log1p(GZ), levels=20, colors="white", linewidths=0.4, alpha=0.6)
    color = colors.get(opt, "#333")
    ax.set_title(opt, fontsize=10, fontweight="semibold", color="#1a1a1a", pad=6)
    ax.tick_params(colors="#555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    line, = ax.plot([], [], color=color, lw=1.8)
    dot,  = ax.plot([], [], "o", color=color, ms=6)
    ax.plot(paths[opt][0][0], paths[opt][0][1], "x", color="#333", ms=8, mew=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    lines[opt] = line
    dots[opt]  = dot

ax4 = axes.flat[3]
ax4.set_facecolor("#f7f7f7")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis("off")
ax4.set_title("iteration counter", fontsize=10, fontweight="semibold", color="#1a1a1a", pad=6)

for i, opt in enumerate(optimizers):
    color = colors.get(opt, "#333")
    y_pos = 0.72 - i * 0.28
    ax4.text(0.08, y_pos + 0.10, opt, fontsize=9, color=color, fontweight="semibold", va="center")
    txt = ax4.text(0.08, y_pos - 0.02, "iter: 0", fontsize=20, color=color, va="center", fontweight="bold")
    counter_texts[opt] = txt

max_frames = max(len(p) for p in paths.values())

def update(frame):
    artists = []
    for opt, path in paths.items():
        i = min(frame, len(path) - 1)
        lines[opt].set_data(path[:i+1, 0], path[:i+1, 1])
        dots[opt].set_data([path[i, 0]], [path[i, 1]])
        counter_texts[opt].set_text(f"iter: {i}")
        artists += [lines[opt], dots[opt], counter_texts[opt]]
    return artists

ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=20, blit=True, repeat=False)

plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.patch.set_facecolor("white")
fig2.suptitle(f"Benchmark Comparison  —  {fn_name}", fontsize=13, fontweight="bold")

metrics = [
    ("iterations", "Iterations to converge"),
    ("time_ms",    "Time elapsed (ms)"),
]

bar_colors = [colors.get(o, "#888") for o in bench["optimizer"]]

for ax, (col, label) in zip(axes2, metrics):
    ax.set_facecolor("#f7f7f7")
    bars = ax.bar(bench["optimizer"], bench[col], color=bar_colors, edgecolor="white", width=0.5)
    ax.set_title(label, fontsize=10, fontweight="semibold", color="#1a1a1a")
    ax.tick_params(colors="#555", labelsize=8)
    ax.set_xticks(range(len(bench["optimizer"])))
    ax.set_xticklabels([o.replace(" ", "\n") for o in bench["optimizer"]], fontsize=7, color="#1a1a1a")
    ax.yaxis.set_tick_params(labelcolor="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                f"{h:.3g}", ha="center", va="bottom", fontsize=8, color="#1a1a1a")

plt.tight_layout()
plt.show()
