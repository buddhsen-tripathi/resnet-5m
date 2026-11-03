"""Compare completed runs from assets/experiments.json; no training dependencies needed.

uv run --no-project --with matplotlib python scripts/plot_experiments.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "assets/experiments.json").read_text())
    runs = data["runs"]
    assert len(runs) == 28 and len({r["run"] for r in runs}) == 28
    assert all(0 < r["parameters"] <= r["total_training_parameters"] <= 5_000_000 for r in runs)
    assert all(0 <= r[key] <= 100 for r in runs for key in ["cifar10_final_percent", "cifar10_flip_percent"])
    assert all(r["kaggle_public"] is None or 0 <= r["kaggle_public"] <= 1 for r in runs)

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 11.6), sharey=True,
                                 gridspec_kw={"width_ratios": [0.75, 1.4, 1.05]}, layout="constrained")
        params, cifar, kaggle = axes
        fig.suptitle("28 models under the 5M budget", fontsize=17, fontweight="bold")
        for y, r in enumerate(runs):
            if r["round"] % 2 == 0 or r["round"] == 9:
                for ax in axes:
                    ax.axhspan(y - 0.48, y + 0.48, color="#edf5f3" if r["round"] == 9 else "#f3f4f5", zorder=0)
            params.text(0.95, y, f'{r["parameters"]:,}', ha="right", va="center")
            cifar.plot(r["cifar10_final_percent"], y, "o", color="#0072B2", markersize=5.5,
                       label="No TTA" if y == 0 else None)
            cifar.plot(r["cifar10_flip_percent"], y, "^", color="#9C4A00", markersize=6,
                       label="Flip TTA" if y == 0 else None)
            score = r["kaggle_public"]
            if score is None:
                kaggle.text(88.5, y, "—", va="center", ha="center", color="#666666")
            else:
                kaggle.plot(100 * score, y, "D", color="#006A52", markersize=5)
                kaggle.text(100 * score + 0.12, y, f"{100 * score:.1f}", va="center", fontsize=10)
        params.set_yticks(range(len(runs)), [f'R{r["round"]}  {r["label"]}' for r in runs])
        params.set(xlim=(0, 1), ylim=(len(runs) - 0.3, -0.7), title="Classifier parameters")
        params.set_xticks([])
        params.tick_params(axis="y", length=0, pad=10)
        for spine in params.spines.values():
            spine.set_visible(False)
        for label, r in zip(params.get_yticklabels(), runs):
            if r["round"] == 9:
                label.set_fontweight("bold")
        cifar.set(xlim=(95.5, 98.0), title="CIFAR-10 test", xlabel="Accuracy (%) · zoomed scale")
        cifar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.065), ncols=2, frameon=False)
        kaggle.set(xlim=(86.8, 90.9), title="Kaggle public", xlabel="Accuracy (%) · zoomed scale")
        for ax in (cifar, kaggle):
            ax.grid(axis="x", color="#dddddd", linewidth=0.7)
            ax.tick_params(axis="y", length=0)
        fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
        fig.text(0.5, 0.022,
                 "Single runs; recipes, seeds and hardware vary. 200 epochs except R8 (300). — = no recorded submission.\n"
                 "Reported final evaluations; no interpolation. R9 trains 4,983,618 parameters including its auxiliary head.",
                 ha="center", va="center", fontsize=10, color="#444444")
        fig.savefig(ROOT / "assets/all-models.png", dpi=180, facecolor="white")
        plt.close(fig)

    lines = ["# All completed models", "", "Generated from [`assets/experiments.json`](../assets/experiments.json) by "
             "[`scripts/plot_experiments.py`](../scripts/plot_experiments.py).", ""]
    lines += [f"- {note}" for note in data["notes"]]
    lines += ["", "| Round | Checkpoint | Classifier parameters | Epochs | CIFAR-10 final (%) | Flip TTA (%) | Kaggle public | Record |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in runs:
        score = "—" if r["kaggle_public"] is None else f'{r["kaggle_public"]:.3f}'
        url = f'https://github.com/buddhsen-tripathi/resnet-5m/releases/download/{r["release"]}/{r["run"]}.pt'
        lines.append(f'| {r["round"]} | [{r["run"]}]({url}) | {r["parameters"]:,} | {r["epochs"]} | '
                     f'{r["cifar10_final_percent"]:.2f} | {r["cifar10_flip_percent"]:.2f} | {score} | '
                     f'[source](../{r["source"]}) |')
    (ROOT / "docs/model-comparison.md").write_text("\n".join(lines) + "\n")
    (ROOT / "assets/all-models-provenance.json").write_text(json.dumps({
        "source": "assets/experiments.json", "script": "scripts/plot_experiments.py",
        "matplotlib": matplotlib.__version__, "runs": len(runs),
        "transformations": ["Kaggle fractions multiplied by 100", "rows retained in round order", "no smoothing or aggregation"],
        "notes": data["notes"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
