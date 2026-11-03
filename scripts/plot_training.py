"""Recreate README curves: uv run --no-project --with matplotlib python scripts/plot_training.py."""

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EPOCH = re.compile(r"epoch (\d+)/(\d+)  loss (\S+)  test_acc (\S+)  lr (\S+)  ([\d.]+)s")


def plot_run(run, directory, expected_epochs, output):
    source = ROOT / "logs" / directory / f"{run}.log"
    rows = EPOCH.findall(source.read_text())
    epochs = [int(row[0]) for row in rows]
    assert epochs == list(range(1, expected_epochs + 1)), "Missing or duplicate epochs"
    loss = [float(row[2]) for row in rows]
    accuracy = [100 * float(row[3]) for row in rows]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        fig.suptitle(f"{run}: recorded training history (seed 1)")
        left.plot(epochs, loss, color="#0072B2", label="Total training loss")
        left.set(xlabel="Epoch", ylabel="Training objective", ylim=(0, None))
        left.legend(frameon=False)
        right.plot(epochs, accuracy, color="#0072B2", label="CIFAR-10 test, no TTA")
        right.plot(epochs[-1], accuracy[-1], "o", color="#0072B2")
        right.set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
        right.legend(frameon=False, loc="lower right")
        for ax in (left, right):
            ax.set_xlim(1, expected_epochs)
            ax.grid(axis="y", alpha=0.2)
        fig.savefig(ROOT / "assets" / output, dpi=180, facecolor="white")
        plt.close(fig)
    return {"run": run, "source": str(source.relative_to(ROOT)), "figure": f"assets/{output}",
            "epochs": len(rows), "transformations": ["accuracy multiplied by 100", "no smoothing"],
            "notes": "Single run; per-epoch mixed-precision evaluation, not final full-precision flip TTA."}


if __name__ == "__main__":
    provenance = [plot_run("R8_base_300", "8-duration", 300, "8-duration.png"),
                  plot_run("R9_combined", "9-combined", 200, "9-combined.png")]
    (ROOT / "assets" / "round8-9-curves.json").write_text(
        json.dumps({"matplotlib": matplotlib.__version__, "figures": provenance}, indent=2) + "\n")
