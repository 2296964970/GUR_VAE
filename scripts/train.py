import os
import random

import numpy as np
import torch

from gru_vae.data import create_normal_loaders
from gru_vae.trainer import OnlineTrainer
from gru_vae.utils import parse_sizes, resolve_device, first_batch_or_exit
from gru_vae.model import OnlineGPVAE
from gru_vae.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    # All parameters now come from config.yaml
    args = load_config()

    set_seed(args.seed)

    # When adaptive_noise_train is enabled, disable dataset noise (sigma=0) and let trainer inject adaptive noise.
    loaders = create_normal_loaders(
        data_dir=args.data_dir,
        case=args.case,
        train_csv=(args.train_csv or None),
        time_length=args.time_length,
        stride=args.stride,
        batch_size=args.batch_size,
        mask_rate=args.mask_rate,
        mask_mode=args.mask_mode,
        block_t_min=args.block_t_min,
        block_t_max=args.block_t_max,
        block_f_min=args.block_f_min,
        block_f_max=args.block_f_max,
        block_max_blocks=args.block_max_blocks,
        corr_t=args.corr_t,
        corr_f=args.corr_f,
        noise_kind=args.noise_kind,
        noise_sigma=(0.0 if args.noise_kind == 'gaussian' else args.noise_sigma),
        noise_bias_min=args.noise_bias_min,
        noise_bias_max=args.noise_bias_max,
        noise_scale_min=args.noise_scale_min,
        noise_scale_max=args.noise_scale_max,
        noise_amp_min=args.noise_amp_min,
        noise_amp_max=args.noise_amp_max,
        seed=args.mask_seed,
    )

    batch0 = first_batch_or_exit(loaders.train, '[error] Train loader is empty (no windows). Adjust time_length/stride.')
    sample_x = batch0[0] if len(batch0) >= 2 else batch0
    T, H = sample_x.shape[1], sample_x.shape[2]

    dec_hidden = parse_sizes(args.dec_hidden)
    model = OnlineGPVAE(
        input_dim=H,
        output_dim=H,
        latent_dim=args.latent_dim,
        enc_hidden_size=args.gru_hidden,
        enc_layers=args.gru_layers,
        dec_hidden=dec_hidden,
        beta=args.beta,
        obs_init_logvar=args.obs_init_logvar,
    )

    device = resolve_device(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = OnlineTrainer(
        model,
        optimizer,
        device=device,
        grad_clip=args.grad_clip,
        beta=args.beta,
        adaptive_noise=True,
    )

    # Enforce output directory layout: output/<case>/models/<exp_name>
    outdir = os.path.join('output', args.case, 'models', args.exp_name)
    os.makedirs(outdir, exist_ok=True)

    train_curve = {'loss': [], 'val': []}
    best_val = float('inf')
    best_state = None

    # Built-in schedules (no extra CLI):
    # - KL warm-up: linear 0 -> beta over first ~20% epochs
    # - Mask curriculum (for block/corr): linear from max(target, 0.4) -> target across training
    warmup_epochs = max(1, int(round(args.epochs * 0.2)))
    start_mask_rate = 0.4 if args.mask_mode in ('block', 'corr', 'window') else args.mask_rate
    for epoch in range(1, args.epochs + 1):
        # Update beta schedule
        cur_beta = args.beta * min(1.0, epoch / float(warmup_epochs))
        trainer.beta = cur_beta
        # Update curriculum mask rate
        if hasattr(loaders.train, 'dataset') and hasattr(loaders.train.dataset, 'mask_rate'):
            if args.mask_mode in ('block', 'corr', 'window'):
                if args.epochs > 1:
                    progress = (epoch - 1) / float(args.epochs - 1)
                else:
                    progress = 1.0
                cur_mask_rate = start_mask_rate + (args.mask_rate - start_mask_rate) * progress
                loaders.train.dataset.mask_rate = float(cur_mask_rate)
        tr = trainer.train_epoch(loaders.train)
        va = trainer.evaluate(loaders.val)
        train_curve['loss'].append(tr.loss)
        train_curve['val'].append(va.loss)
        print(f'Epoch {epoch:03d} | train loss {tr.loss:.4f} (nll {tr.nll:.4f}, kl {tr.kl:.4f}) | '
              f'val loss {va.loss:.4f} (nll {va.nll:.4f}, kl {va.kl:.4f}) | mse_miss val {va.mse_miss:.6f}')
        if va.loss < best_val:
            best_val = va.loss
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': va.loss,
            }

    if best_state is not None:
        torch.save(best_state, os.path.join(outdir, 'ckpt.pt'))

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
