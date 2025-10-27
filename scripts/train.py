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
    )

    device = resolve_device(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    fractions_raw = getattr(args, 'noise_fractions', '')
    if isinstance(fractions_raw, str):
        cleaned = fractions_raw.replace('[', '').replace(']', '')
        parts = [p.strip() for p in cleaned.split(',') if p.strip()]
        noise_fraction_tuple = tuple(float(p) for p in parts)
    elif isinstance(fractions_raw, (list, tuple)):
        noise_fraction_tuple = tuple(float(p) for p in fractions_raw)
    else:
        noise_fraction_tuple = tuple()

    strengths_raw = getattr(args, 'noise_strengths', '')
    if isinstance(strengths_raw, str):
        cleaned_s = strengths_raw.replace('[', '').replace(']', '')
        parts_s = [p.strip() for p in cleaned_s.split(',') if p.strip()]
        noise_strength_tuple = tuple(float(p) for p in parts_s)
    elif isinstance(strengths_raw, (list, tuple)):
        noise_strength_tuple = tuple(float(p) for p in strengths_raw)
    else:
        noise_strength_tuple = tuple()

    trainer = OnlineTrainer(
        model,
        optimizer,
        device=device,
        grad_clip=args.grad_clip,
        beta=args.beta,
        noise_strength=getattr(args, 'noise_strength', 0.5),
        noise_seed=getattr(args, 'noise_seed', 1337),
        noise_fractions=noise_fraction_tuple,
        noise_strengths=noise_strength_tuple,
    )

    # Training output directory: <train_model_root>/<exp_name>
    outdir = args.model_dir
    os.makedirs(outdir, exist_ok=True)

    train_curve = {'loss': [], 'val': []}
    best_val = float('inf')
    best_state = None

    # Built-in schedule (no toggle):
    # - KL warm-up: linear 0 -> beta over first ~20% epochs
    warmup_epochs = max(1, int(round(args.epochs * 0.2)))
    for epoch in range(1, args.epochs + 1):
        # Update beta schedule
        cur_beta = args.beta * min(1.0, epoch / float(warmup_epochs))
        trainer.beta = cur_beta
        tr = trainer.train_epoch(loaders.train)
        va = trainer.evaluate(loaders.val)
        train_curve['loss'].append(tr.loss)
        train_curve['val'].append(va.loss)
        print(
            f'Epoch {epoch:03d} | beta {cur_beta:.4f} | '
            f'train loss {tr.loss:.4f} (nll {tr.nll:.4f}, kl {tr.kl:.4f}) | '
            f'val loss {va.loss:.4f} (nll {va.nll:.4f}, kl {va.kl:.4f}) | '
            f'mse_obs val {va.mse_obs:.6f}'
        )
        if va.loss < best_val:
            best_val = va.loss
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': va.loss,
            }

    if best_state is not None:
        # enrich checkpoint with minimal hyperparams for inference
        best_state.update({
            'latent_dim': args.latent_dim,
            'tcn_channels': args.tcn_channels,
            'tcn_kernel_size': args.tcn_kernel_size,
            'tcn_dropout': args.tcn_dropout,
            'dec_hidden': args.dec_hidden,
        })
        torch.save(best_state, os.path.join(outdir, 'ckpt.pt'))
        # Save slot-wise robust standardization statistics
        np.savez_compressed(
            os.path.join(outdir, 'slot_stats.npz'),
            mean=loaders.slot_mean,
            std=loaders.slot_std,
            slot_kind=str(getattr(loaders, 'slot_kind', 'hour')),
            clip_k=float(getattr(loaders, 'clip_k', 5.0)),
        )

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        va_best = trainer.evaluate(loaders.val)
    else:
        va_best = trainer.evaluate(loaders.val)

    with open(os.path.join(outdir, 'training_curve.tsv'), 'w') as f:
        f.write('\t'.join(map(str, train_curve['loss'])) + '\n')
        f.write('\t'.join(map(str, train_curve['val'])) + '\n')

    print('Training finished. Artifacts saved to:', outdir)


if __name__ == '__main__':
    main()
