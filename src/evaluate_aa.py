import argparse
import csv
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from autoattack import AutoAttack
from omegaconf import OmegaConf

from utils import (
    load_model, load_data, load_label_mappings,
    IMAGENET_MEAN, IMAGENET_STD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"


class PixelModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        out = self.model((x - self.mean) / self.std)
        return out.logits if hasattr(out, "logits") else out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--data", type=Path,
        default=DATA_ROOT / "IMAGES" / "dataset_split_original",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--eps", type=float,
        help="Override budget in units of 1/255, e.g. --eps 4",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.results / "autoattack.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, _ = load_label_mappings()
    checkpoints = sorted(args.results.rglob("best.pt"))
    if not checkpoints:
        parser.error(f"No best.pt found under {args.results}")

    datasets = {"original": args.data / "test"}
    datasets.update({
        f"severity_{s}": args.data / "test_corrupted" / f"severity_{s}"
        for s in (1, 2, 3)
    })
    for path in datasets.values():
        if not path.is_dir():
            parser.error(f"Missing dataset: {path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "test_set", "epsilon", "n",
            "clean_percent", "aa_percent",
        ])

        for checkpoint in checkpoints:
            settings = json.loads(
                checkpoint.with_name("settings.json").read_text()
            )
            cfg = OmegaConf.create(settings["config"])
            eps = args.eps / 255 if args.eps is not None else settings["epsilon"]

            base = load_model(cfg)
            base.load_state_dict(torch.load(
                checkpoint, map_location="cpu", weights_only=True
            ))
            model = PixelModel(base).to(device).eval().requires_grad_(False)

            for name, path in datasets.items():
                is_original = name == "original"

                if is_original:
                    attack = AutoAttack(
                        model, norm="Linf", eps=eps, version="standard",
                        seed=args.seed, device=device, verbose=False,
                    )
                    # Six classes permit only five incorrect target classes.
                    attack.apgd_targeted.n_target_classes = min(9, len(labels) - 1)
                    attack.fab.n_target_classes = min(9, len(labels) - 1)

                data = load_data(path, cfg, labels)
                loader = DataLoader(
                    data, batch_size=args.batch_size, shuffle=False
                )
                clean = robust = total = 0

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                        raise ValueError("load_data must return pixels in [0,1].")

                    with torch.no_grad():
                        correct = model(x).argmax(1).eq(y)
                    clean += correct.sum().item()

                    if is_original:
                        adv = attack.run_standard_evaluation(x, y, bs=len(y))
                        with torch.no_grad():
                            robust += (
                                correct & model(adv).argmax(1).eq(y)
                            ).sum().item()

                    total += len(y)

                clean_percent = 100 * clean / total
                aa_percent = 100 * robust / total if is_original else ""

                writer.writerow([
                    str(checkpoint), name, eps if is_original else "",
                    total, clean_percent, aa_percent,
                ])
                f.flush()

                message = (
                    f"{checkpoint} | {name}: clean={clean_percent:.2f}%"
                )
                if is_original:
                    message += f" AA={aa_percent:.2f}%"
                print(message, flush=True)

            del model, base, attack


if __name__ == "__main__":
    main()