import os
import random

import numpy as np
import torch

from LGSSM_VAE.config import load_config
from LGSSM_VAE.foundation.errors import LGSSMVAEError
from LGSSM_VAE.foundation.utils import first_batch_or_exit, resolve_device
from LGSSM_VAE.data import create_normal_loaders
from LGSSM_VAE.modeling import build_model_from_config
from LGSSM_VAE.pipeline import OnlineTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    try:
        cfg = load_config()

        set_seed(cfg.train.seed)

        loaders = create_normal_loaders(
            train_normal_csv=cfg.train.normal_csv,
            time_length=cfg.window.time_length,
            stride=cfg.window.stride,
            batch_size=cfg.window.batch_size,
            train_ratio=cfg.window.train_ratio,
            val_ratio=cfg.window.val_ratio,
            seed=cfg.train.seed,
            num_workers=cfg.window.num_workers,
            clip_k=cfg.preprocess.clip_k,
            std_floor=cfg.preprocess.std_floor,
        )

        batch0 = first_batch_or_exit(
            loaders.train,
            "[error] Train loader is empty (no windows). Adjust time_length/stride.",
        )
        sample_x = batch0[0] if len(batch0) >= 1 else batch0
        T, H = int(sample_x.shape[1]), int(sample_x.shape[2])

        model, model_name, model_hparams = build_model_from_config(cfg, input_dim=H, time_length=T)

        device = resolve_device(cfg.train.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

        laplace_scales = tuple(float(p) for p in cfg.robust.laplace_scales)
        trainer = OnlineTrainer(
            model,
            optimizer,
            device=device,
            grad_clip=cfg.train.grad_clip,
            beta=cfg.model.beta,
            anchor_lambda=cfg.train.anchor_lambda,
            clean_fraction=cfg.robust.clean_fraction,
            seg_len_min=cfg.robust.seg_len_min,
            seg_len_max=cfg.robust.seg_len_max,
            dims_fraction_min=cfg.robust.dims_fraction_min,
            dims_fraction_max=cfg.robust.dims_fraction_max,
            laplace_scales=laplace_scales,
            rng_seed=cfg.train.seed,
        )

        outdir = cfg.train.model_dir
        os.makedirs(outdir, exist_ok=True)

        train_curve = {"p1_loss": [], "p1_val": [], "p2_loss": [], "p2_val": []}
        best_val = float("inf")
        best_state = None

        p1_epochs = int(cfg.train.phase1_epochs)
        warmup_p1 = max(1, int(round(p1_epochs * float(cfg.train.warmup_frac))))
        for epoch in range(1, p1_epochs + 1):
            cur_beta = float(cfg.model.beta) * min(1.0, epoch / float(warmup_p1))
            trainer.beta = cur_beta
            tr = trainer.train_epoch(loaders.train, robust=False)
            va = trainer.evaluate(loaders.val)
            train_curve["p1_loss"].append(tr.loss)
            train_curve["p1_val"].append(va.loss)
            print(
                f"[P1] Epoch {epoch:03d}/{p1_epochs} | beta {cur_beta:.4f} | "
                f"train {tr.loss:.4f} (NLL {tr.nll:.4f}, KL(total) {tr.kl:.4f}, KL(mean/step used) {tr.kl_used:.4f}) | "
                f"val {va.loss:.4f} (NLL {va.nll:.4f}, KL(total) {va.kl:.4f}, KL(mean/step used) {va.kl_used:.4f}) | mse_obs val {va.mse_obs:.6f}"
            )
            if va.loss < best_val:
                best_val = va.loss
                best_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": va.loss,
                    "phase": "p1",
                    "model_name": model_name,
                    "model_hparams": model_hparams,
                    "time_length": int(T),
                }

        best_val = float("inf")
        best_state = None

        p2_epochs = int(cfg.train.phase2_epochs)
        warmup_p2 = max(1, int(round(p2_epochs * float(cfg.train.warmup_frac))))
        for epoch in range(1, p2_epochs + 1):
            cur_beta = float(cfg.model.beta) * min(1.0, epoch / float(warmup_p2))
            trainer.beta = cur_beta
            tr = trainer.train_epoch(loaders.train, robust=True)
            va = trainer.evaluate(loaders.val)
            train_curve["p2_loss"].append(tr.loss)
            train_curve["p2_val"].append(va.loss)
            print(
                f"[P2] Epoch {epoch:03d}/{p2_epochs} | beta {cur_beta:.4f} | "
                f"train {tr.loss:.4f} (NLL {tr.nll:.4f}, KL(total) {tr.kl:.4f}, KL(mean/step used) {tr.kl_used:.4f}) | "
                f"val {va.loss:.4f} (NLL {va.nll:.4f}, KL(total) {va.kl:.4f}, KL(mean/step used) {va.kl_used:.4f}) | mse_obs val {va.mse_obs:.6f}"
            )
            if va.loss < best_val:
                best_val = va.loss
                best_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": va.loss,
                    "phase": "p2",
                    "model_name": model_name,
                    "model_hparams": model_hparams,
                    "time_length": int(T),
                }

        if best_state is not None:
            torch.save(best_state, os.path.join(outdir, "ckpt.pt"))
            np.savez_compressed(
                os.path.join(outdir, "slot_stats.npz"),
                mean=loaders.slot_mean,
                std=loaders.slot_std,
                clip_k=float(loaders.clip_k),
                std_floor=float(cfg.preprocess.std_floor),
            )

        if best_state is not None:
            model.load_state_dict(best_state["model"])
            va_best = trainer.evaluate(loaders.val)
        else:
            va_best = trainer.evaluate(loaders.val)

        with open(os.path.join(outdir, "training_curve.tsv"), "w") as f:
            f.write("P1_train\t" + "\t".join(map(str, train_curve["p1_loss"])) + "\n")
            f.write("P1_val\t" + "\t".join(map(str, train_curve["p1_val"])) + "\n")
            f.write("P2_train\t" + "\t".join(map(str, train_curve["p2_loss"])) + "\n")
            f.write("P2_val\t" + "\t".join(map(str, train_curve["p2_val"])) + "\n")

        print("Training finished. Artifacts saved to:", outdir)

    except LGSSMVAEError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
