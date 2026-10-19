"""LoRA fine-tune of the SD 1.5 UNet on OpenStax figures, sized for a 6 GB laptop GPU.

Latents and text embeddings are computed once up front, so only the UNet is on the
GPU while training. No horizontal flips: a mirrored heart is wrong anatomy (the
left ventricle would end up on the right)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from ..config import ImagesCfg
from ..energy import StageTracker
from ..runs import RunDir
from .figures import Figure


def caption_for(fig: Figure, trigger: str) -> str:
    alt = fig.alt.removeprefix("This diagram shows ").removeprefix("This figure shows ").rstrip(".")
    return f"{trigger}, medical textbook illustration of {fig.title.lower()}, {alt}"


def train_lora(
    cfg: ImagesCfg,
    figures: list[Figure],
    figures_dir: Path,
    run: RunDir,
    out_dir: Path,
    progress: Callable[[str], None] = print,
) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    lc = cfg.lora
    torch.manual_seed(lc.seed)
    rng = random.Random(lc.seed)
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16, variant="fp16", safety_checker=None, requires_safety_checker=False
    )
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
    vae, text_encoder, tokenizer, unet = pipe.vae.cuda(), pipe.text_encoder.cuda(), pipe.tokenizer, pipe.unet

    with StageTracker(run, "train_lora") as tracker:
        # 1. encode every figure and caption once
        means, stds, embeds, captions = [], [], [], []
        with torch.no_grad():
            for fig in figures:
                img = Image.open(figures_dir / fig.file).convert("RGB")
                x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
                dist = vae.encode(x.half().cuda()).latent_dist
                means.append(dist.mean.cpu())
                stds.append(dist.std.cpu())
                cap = caption_for(fig, cfg.trigger)
                captions.append(cap)
                ids = tokenizer(
                    cap,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                embeds.append(text_encoder(ids.cuda())[0].cpu())
        scaling = vae.config.scaling_factor
        del vae, text_encoder
        pipe.vae = pipe.text_encoder = None
        torch.cuda.empty_cache()

        # 2. LoRA adapters on the attention projections; trainable weights in fp32
        unet = unet.cuda()
        unet.requires_grad_(False)
        unet.add_adapter(
            LoraConfig(r=lc.rank, lora_alpha=lc.alpha, init_lora_weights="gaussian",
                       target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        )  # fmt: skip
        params = [p for p in unet.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        unet.enable_gradient_checkpointing()
        unet.train()
        opt = torch.optim.AdamW(params, lr=lc.learning_rate, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda")
        n_params = sum(p.numel() for p in params)
        progress(f"training {n_params:,} LoRA parameters on {len(figures)} figures for {lc.steps} steps")

        t0 = time.perf_counter()
        window: list[float] = []
        for step in range(1, lc.steps + 1):
            i = rng.randrange(len(figures))
            latents = (means[i] + stds[i] * torch.randn_like(stds[i])).cuda() * scaling
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy.half(), t, embeds[i].cuda()).sample
            loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            window.append(loss.item())
            if step % 50 == 0:
                avg = sum(window) / len(window)
                run.append("train_log.jsonl", {"step": step, "loss": round(avg, 5),
                                               "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
                progress(f"  step {step}/{lc.steps} loss {avg:.4f}")
                window = []
            tracker.tick()

        out_dir.mkdir(parents=True, exist_ok=True)
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(str(out_dir), unet_lora_layers=state, safe_serialization=True)
    meta = {
        "base_model": cfg.base_model,
        "n_figures": len(figures),
        "trainable_params": n_params,
        **lc.model_dump(),
        "captions_example": captions[:3],
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    del unet, pipe
    torch.cuda.empty_cache()
    return meta
