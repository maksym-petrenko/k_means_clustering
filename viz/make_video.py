import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "viz" / "dump"


def build():
    sources = [ROOT / "viz" / "dump.cpp", ROOT / "kmeans.cpp", ROOT / "kmeans.cu", ROOT / "kmeans.h"]
    if DUMP.exists() and all(DUMP.stat().st_mtime > s.stat().st_mtime for s in sources):
        return
    subprocess.run(
        ["nvcc", "-O2", f"-I{ROOT}", *map(str, sources[:3]), "-o", str(DUMP)],
        check=True,
    )


def make_blobs(rng, k, n_per_blob, spread):
    centers = rng.uniform(-10, 10, size=(k, 2))
    return np.concatenate([c + rng.normal(scale=spread, size=(n_per_blob, 2)) for c in centers])


def run_kmeans(points, init, iterations):
    lines = [f"{len(points)} {len(init)} {iterations}"]
    lines += [f"{x} {y}" for x, y in points]
    lines += [f"{x} {y}" for x, y in init]
    out = subprocess.run([str(DUMP)], input="\n".join(lines), capture_output=True, text=True, check=True).stdout
    rows = np.array(out.split(), dtype=float).reshape(-1, 3)
    return rows[:, 1:].reshape(iterations + 1, len(init), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--points", type=int, default=250, help="points per blob")
    ap.add_argument("--spread", type=float, default=1.5)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kmeans.gif")
    args = ap.parse_args()

    build()

    rng = np.random.default_rng(args.seed)
    points = make_blobs(rng, args.k, args.points, args.spread)
    init = points[rng.choice(len(points), args.k, replace=False)]
    history = run_kmeans(points, init, args.iterations)

    colors = plt.cm.tab10(np.arange(args.k) % 10)
    fig, ax = plt.subplots(figsize=(6, 6))
    pad = 1.0
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    ax.set_aspect("equal")

    scatter = ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
    trails = [ax.plot([], [], "-", color=colors[c], lw=1.5)[0] for c in range(args.k)]
    marks = ax.scatter(init[:, 0], init[:, 1], s=200, marker="X", c=colors, edgecolors="black", linewidths=1.5)
    title = ax.set_title("")

    def frame(i):
        centroids = history[i]
        nearest = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1).argmin(1)
        scatter.set_color(colors[nearest])
        for c, trail in enumerate(trails):
            trail.set_data(history[: i + 1, c, 0], history[: i + 1, c, 1])
        marks.set_offsets(centroids)
        title.set_text(f"iteration {i}")
        return [scatter, marks, title, *trails]

    FuncAnimation(fig, frame, frames=len(history)).save(args.out, writer=PillowWriter(fps=args.fps))
    print(args.out)


if __name__ == "__main__":
    main()
