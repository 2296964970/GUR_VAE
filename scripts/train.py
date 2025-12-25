import os
import random

import numpy as np
import torch

from tcn_vae.data import create_normal_loaders
from tcn_vae.trainer import OnlineTrainer
from tcn_vae.utils import parse_sizes, resolve_device, first_batch_or_exit
from tcn_vae.model import TCNVAE
from tcn_vae.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    # All parameters now come from config.yaml
    args = load_config()

    set_seed(args.seed)

    loaders = create_normal_loaders(
        train_normal_csv=args.train_normal_csv,
        time_length=args.time_length,
        stride=args.stride,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=getattr(args, 'num_workers', 0),
        clip_k=getattr(args, 'clip_k', 0.0),
        std_floor=getattr(args, 'std_floor', 1e-3),
    )

    batch0 = first_batch_or_exit(loaders.train, '[error] Train loader is empty (no windows). Adjust time_length/stride.')
    sample_x = batch0[0] if len(batch0) >= 1 else batch0
    T, H = sample_x.shape[1], sample_x.shape[2]

    dec_hidden = parse_sizes(args.dec_hidden)
    tcn_channels = parse_sizes(args.tcn_channels)
    model = TCNVAE(
        input_dim=H,
        output_dim=H,
        latent_dim=args.latent_dim,
        tcn_channels=tcn_channels,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
        dec_hidden=dec_hidden,
        beta=args.beta,
        obs_init_logvar=args.obs_init_logvar,
        enc_diag_eps=getattr(args, 'enc_diag_eps', 1e-4),
        dec_eps=getattr(args, 'dec_eps', 1e-6),
        dec_logvar_min=getattr(args, 'dec_logvar_min', -5.0),
        dec_logvar_max=getattr(args, 'dec_logvar_max', 2.302585092994046),
        prior_rank=getattr(args, 'prior_rank', 4),
        prior_a_init=getattr(args, 'prior_a_init', 0.95),
        prior_q_init=getattr(args, 'prior_q_init', 0.1),
        prior_m0_init=getattr(args, 'prior_m0_init', 0.0),
        prior_P0_init=getattr(args, 'prior_P0_init', 1.0),
        prior_jitter=getattr(args, 'prior_jitter', 1e-6),
        prior_variance_floor=getattr(args, 'prior_variance_floor', 1e-6),
    )

    device = resolve_device(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Robust phase parameters
    lap_scales_raw = getattr(args, 'robust_laplace_scales', '')
    if isinstance(lap_scales_raw, str):
        cleaned = lap_scales_raw.replace('[', '').replace(']', '')
        parts = [p.strip() for p in cleaned.split(',') if p.strip()]
        laplace_scales = tuple(float(p) for p in parts) if parts else (0.1, 0.3, 0.7, 1.2)
    elif isinstance(lap_scales_raw, (list, tuple)):
        laplace_scales = tuple(float(p) for p in lap_scales_raw)
    else:
        laplace_scales = (0.1, 0.3, 0.7, 1.2)

    trainer = OnlineTrainer(
        model,
        optimizer,
        device=device,
        grad_clip=args.grad_clip,
        beta=args.beta,
        anchor_lambda=getattr(args, 'anchor_lambda', 1.0),
        clean_fraction=getattr(args, 'clean_fraction', 0.4),
        seg_len_min=getattr(args, 'seg_len_min', 8),
        seg_len_max=getattr(args, 'seg_len_max', max(8, args.time_length // 2)),
        dims_fraction_min=getattr(args, 'dims_fraction_min', 0.05),
        dims_fraction_max=getattr(args, 'dims_fraction_max', 0.3),
        laplace_scales=laplace_scales,
        rng_seed=getattr(args, 'seed', 1337),
    )

    # Training output directory: <train_model_root>/<exp_name>
    outdir = args.model_dir
    os.makedirs(outdir, exist_ok=True)

    train_curve = {'p1_loss': [], 'p1_val': [], 'p2_loss': [], 'p2_val': []}
    best_val = float('inf')
    best_state = None

    # Phase-1: identity mapping (no augmentation, ELBO only)
    p1_epochs = int(getattr(args, 'phase1_epochs', max(1, args.epochs // 4)))
    warmup_p1 = max(1, int(round(p1_epochs * float(getattr(args, 'warmup_frac', 0.2)))))
    for epoch in range(1, p1_epochs + 1):
        cur_beta = args.beta * min(1.0, epoch / float(warmup_p1))
        trainer.beta = cur_beta
        tr = trainer.train_epoch(loaders.train, robust=False)
        va = trainer.evaluate(loaders.val)
        train_curve['p1_loss'].append(tr.loss)
        train_curve['p1_val'].append(va.loss)
        print(
            f'[P1] Epoch {epoch:03d}/{p1_epochs} | beta {cur_beta:.4f} | '
            f'train {tr.loss:.4f} (nll {tr.nll:.4f}, kl {tr.kl:.4f}) | '
            f'val {va.loss:.4f} (nll {va.nll:.4f}, kl {va.kl:.4f}) | mse_obs val {va.mse_obs:.6f}'
        )
        if va.loss < best_val:
            best_val = va.loss
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': va.loss,
                'phase': 'p1',
            }

    # Phase-2: robust training with Laplace segment attacks + anchor loss
    p2_epochs = int(getattr(args, 'phase2_epochs', max(1, args.epochs - p1_epochs)))
    warmup_p2 = max(1, int(round(p2_epochs * float(getattr(args, 'warmup_frac', 0.2)))))
    for epoch in range(1, p2_epochs + 1):
        cur_beta = args.beta * min(1.0, epoch / float(warmup_p2))
        trainer.beta = cur_beta
        tr = trainer.train_epoch(loaders.train, robust=True)
        va = trainer.evaluate(loaders.val)
        train_curve['p2_loss'].append(tr.loss)
        train_curve['p2_val'].append(va.loss)
        print(
            f'[P2] Epoch {epoch:03d}/{p2_epochs} | beta {cur_beta:.4f} | '
            f'train {tr.loss:.4f} (nll {tr.nll:.4f}, kl {tr.kl:.4f}) | '
            f'val {va.loss:.4f} (nll {va.nll:.4f}, kl {va.kl:.4f}) | mse_obs val {va.mse_obs:.6f}'
        )
        if va.loss < best_val:
            best_val = va.loss
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': va.loss,
                'phase': 'p2',
            }

    if best_state is not None:
        # enrich checkpoint with minimal hyperparams for inference
        best_state.update({
            'latent_dim': args.latent_dim,
            'tcn_channels': args.tcn_channels,
            'tcn_kernel_size': args.tcn_kernel_size,
            'tcn_dropout': args.tcn_dropout,
            'dec_hidden': args.dec_hidden,
            'enc_diag_eps': getattr(args, 'enc_diag_eps', 1e-4),
            'dec_eps': getattr(args, 'dec_eps', 1e-6),
            'dec_logvar_min': getattr(args, 'dec_logvar_min', -5.0),
            'dec_logvar_max': getattr(args, 'dec_logvar_max', 2.302585092994046),
            'prior_rank': getattr(args, 'prior_rank', 4),
            'prior_a_init': getattr(args, 'prior_a_init', 0.95),
            'prior_q_init': getattr(args, 'prior_q_init', 0.1),
            'prior_m0_init': getattr(args, 'prior_m0_init', 0.0),
            'prior_P0_init': getattr(args, 'prior_P0_init', 1.0),
            'prior_jitter': getattr(args, 'prior_jitter', 1e-6),
            'prior_variance_floor': getattr(args, 'prior_variance_floor', 1e-6),
        })
        torch.save(best_state, os.path.join(outdir, 'ckpt.pt'))
        # Save slot-wise robust stats for inference (5-minute within-day slots).
        np.savez_compressed(
            os.path.join(outdir, 'slot_stats.npz'),
            mean=loaders.slot_mean,
            std=loaders.slot_std,
            clip_k=float(getattr(loaders, 'clip_k', 5.0)),
            std_floor=float(getattr(args, 'std_floor', 1e-3)),
        )

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        va_best = trainer.evaluate(loaders.val)
    else:
        va_best = trainer.evaluate(loaders.val)

    with open(os.path.join(outdir, 'training_curve.tsv'), 'w') as f:
        f.write('P1_train\t' + '\t'.join(map(str, train_curve['p1_loss'])) + '\n')
        f.write('P1_val\t' + '\t'.join(map(str, train_curve['p1_val'])) + '\n')
        f.write('P2_train\t' + '\t'.join(map(str, train_curve['p2_loss'])) + '\n')
        f.write('P2_val\t' + '\t'.join(map(str, train_curve['p2_val'])) + '\n')

    print('Training finished. Artifacts saved to:', outdir)


if __name__ == '__main__':
    main()
