import argparse
import os
import random

import numpy as np
import torch

from gru_vae.data import create_normal_loaders
from gru_vae.trainer import OnlineTrainer
from gru_vae.utils import parse_sizes, resolve_device, first_batch_or_exit
from gru_vae.model import OnlineGPVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--case', type=str, default='case14')
    p.add_argument('--time_length', type=int, default=96)
    p.add_argument('--stride', type=int, default=48)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mask_rate', type=float, default=0.3)
    p.add_argument('--mask_mode', type=str, default='iid', choices=['iid', 'block', 'corr'])
    # Block mask params
    p.add_argument('--block_t_min', type=int, default=2)
    p.add_argument('--block_t_max', type=int, default=8)
    p.add_argument('--block_f_min', type=int, default=4)
    p.add_argument('--block_f_max', type=int, default=32)
    p.add_argument('--block_max_blocks', type=int, default=4)
    # Correlated mask params
    p.add_argument('--corr_t', type=int, default=7)
    p.add_argument('--corr_f', type=int, default=15)
    p.add_argument('--mask_seed', type=int, default=1337)
    # Model (GRU-only)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--dec_hidden', type=str, default='256,256')
    p.add_argument('--gru_hidden', type=int, default=256)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--beta', type=float, default=0.1)
    # Observation variance
    p.add_argument('--obs_learn_var', dest='obs_learn_var', action='store_true', help='Learn observation variance (default)')
    p.add_argument('--no-obs_learn_var', dest='obs_learn_var', action='store_false', help='Use fixed observation variance')
    p.set_defaults(obs_learn_var=True)
    p.add_argument('--obs_init_logvar', type=float, default=-3.5)
    p.add_argument('--grad_clip', type=float, default=1e4)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    # Train
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--exp_name', type=str, default='gru_base_ep40')
    p.add_argument('--save_dir', type=str, default='models')
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()

    set_seed(args.seed)

    loaders = create_normal_loaders(
        data_dir=args.data_dir,
        case=args.case,
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
        obs_learn_var=args.obs_learn_var,
        obs_init_logvar=args.obs_init_logvar,
    )

    device = resolve_device(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = OnlineTrainer(model, optimizer, device=device, grad_clip=args.grad_clip, beta=args.beta)

    outdir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(outdir, exist_ok=True)

    train_curve = {'loss': [], 'val': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
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
