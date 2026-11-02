import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import Flux2Pipeline
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"


MODEL_ID = "diffusers/FLUX.2-dev-bnb-4bit"
SPLIT_ROOT = DATA_ROOT / "IMAGES" / "dataset_split_original"
PROMPT_DIR = DATA_ROOT / "original_image_descriptions"
CACHE_PATH = DATA_ROOT / "models_cache"
OUTPUT_DIR = PROJECT_ROOT / "lora-railway-obstacles-flux2-final"

NUM_EPOCHS = 40
SAVE_EPOCHS = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40}
BATCH_SIZE = 1
GRAD_ACCUM = 4
SEED = 0

TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj",
    "add_v_proj", "to_add_out",
]


def prepare_metadata(split_dir):
    """Match split images to captions in PROMPT_DIR."""
    images = sorted(
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No images found in {split_dir}")

    with (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for image in images:
            caption_file = PROMPT_DIR / f"{image.stem}.txt"
            if not caption_file.is_file():
                raise FileNotFoundError(caption_file)

            caption = caption_file.read_text(encoding="utf-8").strip()
            if not caption:
                raise ValueError(f"Empty caption: {caption_file}")

            f.write(json.dumps({
                "file_name": image.relative_to(split_dir).as_posix(),
                "text": caption,
            }, ensure_ascii=False) + "\n")

    return {image.stem for image in images}


def encode_split(pipe, path, device, dtype):
    transform = transforms.Compose([
        transforms.Resize(
            (1088, 1920),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = load_dataset(
        "imagefolder", data_dir=str(path), split="train"
    )
    encoded = []

    for item in tqdm(dataset, desc=f"Encoding {path.name}"):
        image = transform(item["image"].convert("RGB"))
        image = image.unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            latents = pipe._encode_vae_image(
                image=image, generator=None
            )
            packed = pipe._pack_latents(latents)
            img_ids = pipe._prepare_latent_ids(latents)

            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=item["text"],
                max_sequence_length=512,
            )

        encoded.append({
            "latents": packed[0].cpu(),
            "prompt_embeds": prompt_embeds[0].cpu(),
            "text_ids": (
                text_ids[0] if text_ids.ndim == 3 else text_ids
            ).cpu(),
            "img_ids": (
                img_ids[0] if img_ids.ndim == 3 else img_ids
            ).cpu(),
        })

    return encoded


def flow_loss(transformer, batch, device, dtype, generator=None):
    latents = batch["latents"].to(device)
    prompt_embeds = batch["prompt_embeds"].to(device)
    text_ids = batch["text_ids"].to(device)
    img_ids = batch["img_ids"].to(device)

    bsz = latents.shape[0]

    noise = torch.randn(
        latents.shape,
        device=device,
        dtype=latents.dtype,
        generator=generator,
    )
    sigma = torch.rand(
        (bsz,),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    noisy_latents = (
        (1 - sigma[:, None, None]) * latents
        + sigma[:, None, None] * noise
    )
    target = noise - latents

    with torch.autocast("cuda", dtype=dtype):
        prediction = transformer(
            hidden_states=noisy_latents,
            timestep=sigma,
            guidance=torch.full((bsz,), 4.0, device=device),
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids[0],
            img_ids=img_ids[0],
            return_dict=False,
        )[0]

    return F.mse_loss(prediction.float(), target.float())


@torch.no_grad()
def validate(transformer, val_data, device, dtype):
    transformer.eval()
    generator = torch.Generator(device=device).manual_seed(
        10_000 + SEED
    )

    losses = [
        flow_loss(transformer, batch, device, dtype, generator).item()
        for batch in DataLoader(val_data, batch_size=1)
    ]
    return sum(losses) / len(losses)


def main():
    learning_rate = 1.79371796e-05
    rank = 8
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda"
    dtype = torch.bfloat16
    train_path = SPLIT_ROOT / "train"
    val_path = SPLIT_ROOT / "val"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = prepare_metadata(train_path)
    val_ids = prepare_metadata(val_path)
    if train_ids & val_ids:
        raise ValueError("Training and validation image IDs overlap.")

    print(f"Using HPO result: lr={learning_rate}, r={rank}")

    pipe = Flux2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_PATH,
    )
    pipe.vae.to(device)
    pipe.text_encoder.to(device)
    pipe.transformer.to("cpu")

    train_data = encode_split(pipe, train_path, device, dtype)
    val_data = encode_split(pipe, val_path, device, dtype)

    del pipe.vae, pipe.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    transformer = pipe.transformer.to(device)
    transformer.requires_grad_(False)
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=TARGET_MODULES,
            init_lora_weights=True,
        ),
    )
    transformer.enable_gradient_checkpointing()

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    history = []

    for epoch in range(NUM_EPOCHS):
        transformer.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        for step, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        ):
            window_start = (step // GRAD_ACCUM) * GRAD_ACCUM
            window_size = min(
                GRAD_ACCUM,
                len(loader) - window_start,
            )

            loss = flow_loss(transformer, batch, device, dtype)
            (loss / window_size).backward()
            train_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss = validate(transformer, val_data, device, dtype)
        train_loss /= len(loader)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        print(record)

        (OUTPUT_DIR / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if epoch + 1 in SAVE_EPOCHS:
            checkpoint = OUTPUT_DIR / f"epoch_{epoch + 1}"
            transformer.save_pretrained(checkpoint)
            print(f"Saved adapter: {checkpoint}")


if __name__ == "__main__":
    main()