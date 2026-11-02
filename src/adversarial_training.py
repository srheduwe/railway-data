import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from trades import trades_loss
from utils import load_model, load_data, load_label_mappings, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"


@dataclass
class TrainConfig:
    model: str = "cnx"
    augmentation: str | None = None
    seed: int = 22  # controls model initialization, training randomness, and sampling batches.
    data_root: str = str(DATA_ROOT / "IMAGES")
    config_path: str | None = None
    output_path: str | None = None
    epsilon: float = 8 / 255  # Linf perturbation bound
    pool_size: int = 10000 # Number of additional images available for sampling —not the number placed in every batch
    real_fraction: float = 0.3     # Fraction of each mixed batch from original data.
    data_seed: int = 123  # which additional files enter the pool
    batch_size: int = 32
    epochs: int = 1000
    beta: float = 5.0     # beta is a hyperparameter that controls the trade-off between natural accuracy and robustness
    lr: float = 0.0003
    step_size: float = 2 / 255
    ema_decay: float = 0.995
    val_every: int = 1  # Validate every val_every epochs.
    weight_decay: float = 2e-4
    trades_steps: int = 10 # Number of steps for the inner maximization in TRADES loss.
    pgd_steps: int = 40 # Number of steps for the PGD attack used in validation.
    common_augmentation: bool = False

class PixelModel(nn.Module):
    """Accept [0,1] pixels; apply ImageNet normalization inside the model."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        output = self.model((x - self.mean) / self.std)
        return output.logits if hasattr(output, "logits") else output


def select_pool(data, count, num_classes, data_seed):
    """Select a reproducible, approximately class-balanced image pool."""
    generator = torch.Generator().manual_seed(data_seed)
    selected = []

    for label in range(num_classes):
        indices = [i for i, target in enumerate(data.targets) if target == label]
        quota = count // num_classes + (label < count % num_classes)

        if len(indices) < quota:
            raise ValueError(
                f"Class {label}: requested {quota} additional images, "
                f"found {len(indices)}."
            )

        order = torch.randperm(len(indices), generator=generator)
        selected.extend(indices[i] for i in order[:quota].tolist())

    return selected


def draw(data, indices, count, generator):
    choices = torch.randint(len(indices), (count,), generator=generator)
    return [data[indices[i]] for i in choices.tolist()]


def validate(model, loader, epsilon, step_size, pgd_steps):
    """Clean accuracy and white-box PGD robust accuracy on real validation data."""
    model.eval()
    clean = robust = total = 0

    # Make validation attacks repeatable without changing training randomness.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(10_000)
        torch.cuda.manual_seed_all(10_000)

        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                survived = model(x).argmax(1).eq(y)
                clean += survived.sum().item()

            adv = (
                x + torch.empty_like(x).uniform_(-epsilon, epsilon)
            ).clamp(0, 1)

            for _ in range(pgd_steps):
                adv = adv.detach().requires_grad_(True)
                logits = model(adv)

                # correctly classified before and after the attack
                survived &= logits.detach().argmax(1).eq(y)

                gradient = torch.autograd.grad(
                    F.cross_entropy(logits, y), adv
                )[0]

                adv = adv.detach() + step_size * gradient.sign()
                adv = torch.maximum(
                    torch.minimum(adv, x + epsilon), x - epsilon
                ).clamp(0, 1)

            with torch.no_grad():
                survived &= model(adv).argmax(1).eq(y)

            robust += survived.sum().item()
            total += len(y)

    return clean / total, robust / total


def main(c: TrainConfig | None = None):
    c = c if c is not None else TrainConfig()
    print(f"Running {c.model} with augmentation={c.augmentation}, seed={c.seed}, epochs={c.epochs}, batch_size={c.batch_size}, lr={c.lr}, beta={c.beta}, pool_size={c.pool_size}, real_fraction={c.real_fraction}")

    torch.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)

    data_root = Path(c.data_root)
    original_path = data_root / "dataset_split_original"

    additional_path = (
        data_root / f"dataset_split_{c.augmentation}" / "train"
        if c.augmentation is not None else None
    )

    augmentation_name = c.augmentation or "original"
    pool_size = c.pool_size if additional_path is not None else 0
    output_path = Path(
        c.output_path
        or PROJECT_ROOT / "results" / str(round(c.epsilon, 2)) / c.model
        / augmentation_name / f"pool{pool_size}" / f"seed{c.seed}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(
        c.config_path or PROJECT_ROOT / "config" / "model" / f"{c.model}.yaml"
    )
    label2idx, _ = load_label_mappings()
    num_classes = 6

    if (
        len(label2idx) != num_classes
        or set(label2idx.values()) != set(range(num_classes))
    ):
        raise ValueError("Label mapping must contain indices 0..C-1.")

    real = load_data(original_path / "train", cfg, label2idx, augment=c.common_augmentation)
    val = load_data(original_path / "val", cfg, label2idx, augment=False)
    additional = (
        load_data(additional_path, cfg, label2idx, augment=c.common_augmentation)
        if additional_path is not None else None
    )
    pool = (
        select_pool(additional, c.pool_size, num_classes, c.data_seed)
        if additional is not None else []
    )

    real_count = (
        round(c.batch_size * c.real_fraction)
        if additional is not None else c.batch_size
    )
    if additional is not None and not 0 < real_count < c.batch_size:
        raise ValueError("A mixed batch must contain both data sources.")

    model = PixelModel(load_model(cfg)).cuda()

    # weights are an exponentially weighted average of the training model’s weights:
    averaged = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=c.lr,
        weight_decay=c.weight_decay,
        decoupled_weight_decay=True
    )

    steps_per_epoch = math.ceil(len(real) / c.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=c.epochs * steps_per_epoch
    )

    val_loader = DataLoader(val, batch_size=c.batch_size, shuffle=False)
    draw_generator = torch.Generator().manual_seed(c.seed)
    real_indices = list(range(len(real)))

    settings = {
        **asdict(c),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "output_path": str(output_path),
        "original_images": len(real),
        "additional_pool_size": len(pool),
        "additional_paths": (
            [additional.samples[i][0] for i in pool]
            if additional is not None else []
        ),
        "real_fraction_actual": real_count / c.batch_size,
        "steps_per_epoch": steps_per_epoch,
    }
    (output_path / "settings.json").write_text(json.dumps(settings, indent=2))

    history, best_robust = [], -1.0

    for epoch in range(1, c.epochs + 1):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            examples = draw(
                real, real_indices, real_count, draw_generator
            )
            if additional is not None:
                examples += draw(
                    additional, pool,
                    c.batch_size - real_count, draw_generator
                )

            x, y = zip(*examples)
            x = torch.stack(x).cuda()
            y = torch.tensor(y, device="cuda")

            # Unmodified trades_loss() handles attack generation and
            # optimizer.zero_grad(); we handle backward and the update.
            loss = trades_loss(
                model=model,
                x_natural=x,
                y=y,
                optimizer=optimizer,
                step_size=c.step_size,
                epsilon=c.epsilon,
                perturb_steps=c.trades_steps,
                beta=c.beta,
                distance="l_inf",
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                for avg, current in zip(
                    averaged.parameters(), model.parameters()
                ):
                    avg.lerp_(current, 1 - c.ema_decay)
                for avg, current in zip(
                    averaged.buffers(), model.buffers()
                ):
                    avg.copy_(current)

        if epoch % c.val_every:
            continue

        clean, robust = validate(
            averaged, val_loader, c.epsilon, c.step_size, c.pgd_steps
        )
        record = {
            "epoch": epoch,
            "loss": epoch_loss / steps_per_epoch,
            "clean_val": clean,
            "robust_val": robust,
        }
        history.append(record)
        print(record, flush=True)

        (output_path / "history.json").write_text(
            json.dumps(history, indent=2)
        )

        if robust > best_robust:
            best_robust = robust
            torch.save(averaged.model.state_dict(), output_path / "best.pt")

    torch.save(averaged.model.state_dict(), output_path / "last.pt")
    return str(output_path)


if __name__ == "__main__":
    main()