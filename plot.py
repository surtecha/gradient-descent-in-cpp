import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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
    "vanilla gradient descent":      "#00ffe0",
    "gradient descent + momentum":   "#ffdd00",
    "nesterov momentum":             "#ff6ec7",
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
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle(f"Gradient Descent Trajectories  ·  {fn_name}", color="white", fontsize=15, y=1.01)

lines = {}
dots  = {}

for ax, opt in zip(axes.flat, optimizers):
    ax.set_facecolor("#1a1a2e")
    ax.contourf(GX, GY, np.log1p(GZ), levels=40, cmap="magma", alpha=0.85)
    ax.contour( GX, GY, np.log1p(GZ), levels=20, colors="white", linewidths=0.3, alpha=0.3)
    color = colors.get(opt, "white")
    ax.set_title(opt, color=color, fontsize=10, pad=6)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    line, = ax.plot([], [], color=color, lw=1.5, alpha=0.95)
    dot,  = ax.plot([], [], "o", color=color, ms=6)
    ax.plot(paths[opt][0][0], paths[opt][0][1], "x", color="white", ms=8, mew=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    lines[opt] = line
    dots[opt]  = dot

for ax in axes.flat[len(optimizers):]:
    ax.set_visible(False)

max_frames = max(len(p) for p in paths.values())

def animate(frame):
    for opt, path in paths.items():
        i = min(frame, len(path) - 1)
        lines[opt].set_data(path[:i+1, 0], path[:i+1, 1])
        dots[opt].set_data([path[i, 0]], [path[i, 1]])
    return list(lines.values()) + list(dots.values())

ani = animation.FuncAnimation(fig, animate, frames=max_frames, interval=20, blit=True, repeat=False)

plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.patch.set_facecolor("#0f0f0f")
fig2.suptitle(f"Benchmark Comparison  ·  {fn_name}", color="white", fontsize=14)

metrics = [
    ("iterations", "Iterations to converge"),
    ("time_ms",    "Time elapsed (ms)"),
]

bar_colors = [colors.get(o, "#aaa") for o in bench["optimizer"]]

for ax, (col, label) in zip(axes2, metrics):
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(bench["optimizer"], bench[col], color=bar_colors, edgecolor="#333", width=0.5)
    ax.set_title(label, color="white", fontsize=10)
    ax.tick_params(colors="gray")
    ax.set_xticks(range(len(bench["optimizer"])))
    ax.set_xticklabels([o.replace(" ", "\n") for o in bench["optimizer"]], color="white", fontsize=7)
    ax.yaxis.set_tick_params(labelcolor="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                f"{h:.3g}", ha="center", va="bottom", color="white", fontsize=8)

plt.tight_layout()
plt.show()
