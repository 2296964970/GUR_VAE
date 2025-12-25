"""诊断后验塌缩（Posterior Collapse）问题。

检查项目:
1. KL 散度统计：时间平均、样本平均，判断是否趋近于零
2. 干预实验：比较正常 z、随机 z、时间打乱 z 的重构误差

用法:
    python scripts/diagnose_collapse.py
"""
from __future__ import annotations

import torch
import numpy as np

from tcn_vae.config import load_config
from tcn_vae.data import create_normal_loaders
from tcn_vae.model import TCNVAE
from tcn_vae.utils import parse_sizes, resolve_device


def load_model_from_ckpt(ckpt_path: str, input_dim: int, device: torch.device) -> TCNVAE:
    """从 checkpoint 加载模型。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TCNVAE(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=ckpt['latent_dim'],
        tcn_channels=parse_sizes(ckpt['tcn_channels']),
        tcn_kernel_size=ckpt['tcn_kernel_size'],
        tcn_dropout=ckpt['tcn_dropout'],
        dec_hidden=parse_sizes(ckpt['dec_hidden']),
        enc_diag_eps=ckpt.get('enc_diag_eps', 1e-4),
        dec_eps=ckpt.get('dec_eps', 1e-6),
        dec_logvar_min=ckpt.get('dec_logvar_min', -5.0),
        dec_logvar_max=ckpt.get('dec_logvar_max', 2.302585092994046),
        prior_rank=ckpt.get('prior_rank', 4),
        prior_a_init=ckpt.get('prior_a_init', 0.95),
        prior_q_init=ckpt.get('prior_q_init', 0.1),
        prior_m0_init=ckpt.get('prior_m0_init', 0.0),
        prior_P0_init=ckpt.get('prior_P0_init', 1.0),
        prior_jitter=ckpt.get('prior_jitter', 1e-6),
        prior_variance_floor=ckpt.get('prior_variance_floor', 1e-6),
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model


def compute_kl_per_timestep(
    model: TCNVAE,
    mu: torch.Tensor,
    chol: torch.Tensor,
) -> torch.Tensor:
    """计算每个时间步的 KL 散度。

    Returns: [T] 每个时间步的平均 KL
    """
    B, D, T = mu.shape
    device, dtype = mu.device, mu.dtype
    A, Q, m0, P0 = model.prior.transition_matrices(device=device, dtype=dtype)
    jitter_eye = model.prior.jitter * torch.eye(D, device=device, dtype=dtype)

    m_prev = m0.unsqueeze(0).expand(B, -1).contiguous()
    P_prev = P0.unsqueeze(0).expand(B, -1, -1).contiguous()

    kl_per_t = []
    for t in range(T):
        m_pred, P_pred = model.prior.predict(m_prev, P_prev, A=A, Q=Q)
        mu_t = mu[:, :, t]
        chol_t = chol[:, t, :, :]
        kl_t = model.prior.kl_q_prior(mu_t, chol_t, m_pred, P_pred)
        kl_per_t.append(kl_t.mean().item())
        # 更新状态
        m_prev = mu_t.detach()
        Sigma_t = torch.matmul(chol_t, chol_t.transpose(-1, -2)) + jitter_eye
        P_prev = Sigma_t.detach()

    return torch.tensor(kl_per_t)


@torch.no_grad()
def diagnose_kl(model: TCNVAE, loader, device: torch.device, max_batches: int = 50):
    """诊断 KL 散度分布。"""
    print("\n" + "=" * 60)
    print("【诊断 1】KL 散度统计")
    print("=" * 60)

    all_kl_sum = []       # 每个样本的 KL 总和
    all_kl_per_t = []     # 每个时间步的 KL（所有 batch 累积）

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, m = batch
        x = x.float().to(device)
        m = m.float().to(device)

        mu, chol = model._encode(x, m)

        # 计算总 KL
        kl_sum = model._kl_over_time(mu, chol)  # [B]
        all_kl_sum.extend(kl_sum.cpu().numpy().tolist())

        # 计算每时间步 KL
        kl_t = compute_kl_per_timestep(model, mu, chol)
        all_kl_per_t.append(kl_t)

    all_kl_sum = np.array(all_kl_sum)
    all_kl_per_t = torch.stack(all_kl_per_t).mean(dim=0).numpy()

    T = len(all_kl_per_t)
    D = model.latent_dim

    print(f"\n样本数: {len(all_kl_sum)}, 时间步: {T}, 潜在维度: {D}")
    print(f"\n【样本级 KL 统计】(sum over time)")
    print(f"  Mean:   {all_kl_sum.mean():.4f}")
    print(f"  Std:    {all_kl_sum.std():.4f}")
    print(f"  Min:    {all_kl_sum.min():.4f}")
    print(f"  Max:    {all_kl_sum.max():.4f}")
    print(f"  Median: {np.median(all_kl_sum):.4f}")

    # 每时间步平均 KL
    print(f"\n【时间步级 KL 统计】(mean over samples)")
    print(f"  Mean:   {all_kl_per_t.mean():.4f}")
    print(f"  Std:    {all_kl_per_t.std():.4f}")
    print(f"  Min:    {all_kl_per_t.min():.4f} (t={all_kl_per_t.argmin()})")
    print(f"  Max:    {all_kl_per_t.max():.4f} (t={all_kl_per_t.argmax()})")

    # 参考值
    expected_kl_per_step = all_kl_sum.mean() / T
    print(f"\n【参考】")
    print(f"  每步平均 KL: {expected_kl_per_step:.4f}")
    print(f"  每维度平均 KL: {expected_kl_per_step / D:.6f}")

    # 判断
    print(f"\n【诊断结论】")
    if all_kl_sum.mean() < 1.0:
        print("  ⚠️  警告: KL 总和 < 1.0，存在严重后验塌缩风险!")
    elif all_kl_sum.mean() < D * 0.1:
        print(f"  ⚠️  警告: KL 偏低 (< {D * 0.1:.1f})，可能存在轻度后验塌缩")
    else:
        print(f"  ✓  KL 值正常 (>= {D * 0.1:.1f})，后验塌缩风险较低")

    return all_kl_sum.mean()


@torch.no_grad()
def diagnose_intervention(model: TCNVAE, loader, device: torch.device, max_batches: int = 20):
    """干预实验：比较不同 z 的重构误差。"""
    print("\n" + "=" * 60)
    print("【诊断 2】干预实验 (Intervention Test)")
    print("=" * 60)

    mse_normal = []    # 正常 z
    mse_random = []    # 随机高斯噪声 z
    mse_shuffle = []   # 时间打乱 z
    mse_zero = []      # 全零 z

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, m = batch
        x = x.float().to(device)
        m = m.float().to(device)
        B, T, H = x.shape

        # 编码得到正常 z
        mu, chol = model._encode(x, m)  # mu: [B, D, T]
        z_normal = mu

        # 生成干预 z
        z_random = torch.randn_like(mu)

        # 时间打乱：对每个样本独立打乱时间维度
        perm = torch.randperm(T, device=device)
        z_shuffle = mu[:, :, perm]

        # 全零 z
        z_zero = torch.zeros_like(mu)

        # 解码并计算 MSE (只在观测位置)
        def compute_mse(z):
            recon, _ = model.decoder(z)
            diff = (recon - x) ** 2 * m
            return (diff.sum() / m.sum().clamp_min(1)).item()

        mse_normal.append(compute_mse(z_normal))
        mse_random.append(compute_mse(z_random))
        mse_shuffle.append(compute_mse(z_shuffle))
        mse_zero.append(compute_mse(z_zero))

    mse_normal = np.mean(mse_normal)
    mse_random = np.mean(mse_random)
    mse_shuffle = np.mean(mse_shuffle)
    mse_zero = np.mean(mse_zero)

    print(f"\n【重构 MSE 对比】")
    print(f"  正常 z (posterior mean):  {mse_normal:.6f} (baseline)")
    print(f"  随机 z (standard normal): {mse_random:.6f} (ratio: {mse_random / mse_normal:.2f}x)")
    print(f"  时间打乱 z:               {mse_shuffle:.6f} (ratio: {mse_shuffle / mse_normal:.2f}x)")
    print(f"  全零 z:                   {mse_zero:.6f} (ratio: {mse_zero / mse_normal:.2f}x)")

    print(f"\n【诊断结论】")
    # 如果 z 携带信息，打乱或替换后误差应显著增加
    ratio_random = mse_random / mse_normal
    ratio_shuffle = mse_shuffle / mse_normal

    if ratio_random < 1.5 and ratio_shuffle < 1.5:
        print("  ⚠️  警告: 干预后 MSE 变化不显著 (<1.5x)，解码器可能不依赖 z!")
        print("       这表明存在后验塌缩，潜在变量未携带有效信息。")
    elif ratio_random < 2.0:
        print("  ⚠️  注意: 随机 z 的 MSE 仅增加 {:.1f}x，z 的信息量可能偏低".format(ratio_random))
    else:
        print("  ✓  干预测试通过，潜在变量 z 携带有效信息")
        print(f"     随机替换后 MSE 增加 {ratio_random:.1f}x，时间打乱后增加 {ratio_shuffle:.1f}x")

    return ratio_random, ratio_shuffle


@torch.no_grad()
def diagnose_posterior_variance(model: TCNVAE, loader, device: torch.device, max_batches: int = 20):
    """诊断 posterior 方差是否接近 prior 预测方差（塌缩风险）。"""
    print("\n" + "=" * 60)
    print("【诊断 3】后验方差分析")
    print("=" * 60)

    all_posterior_var = []
    T_ref = None

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, m = batch
        x = x.float().to(device)
        m = m.float().to(device)

        mu, chol = model._encode(x, m)  # mu: [B, D, T], chol: [B, T, D, D]
        if T_ref is None:
            T_ref = int(mu.shape[2])
        # 计算后验方差 (对角线)
        var = (chol ** 2).sum(dim=-1)  # [B, T, D] 每个维度的方差
        all_posterior_var.append(var.mean(dim=(0, 1)).cpu())  # [D]

    posterior_var = torch.stack(all_posterior_var).mean(dim=0).numpy()  # [D]

    # Compute unconditional prior predictive marginal variances over T steps.
    if T_ref is None:
        print("  [warn] No batches processed; skip posterior variance diagnosis.")
        return
    A, Q, m0, P0 = model.prior.transition_matrices(device=device, dtype=torch.float32)
    m_prev = m0.view(1, -1).expand(1, -1).contiguous()
    P_prev = P0.unsqueeze(0).expand(1, -1, -1).contiguous()
    prior_vars = []
    for _t in range(T_ref):
        m_pred, P_pred = model.prior.predict(m_prev, P_prev, A=A, Q=Q)
        prior_vars.append(torch.diagonal(P_pred[0], dim1=-2, dim2=-1).cpu())
        m_prev, P_prev = m_pred.detach(), P_pred.detach()
    prior_var = torch.stack(prior_vars, dim=0).mean(dim=0).numpy()  # [D]

    ratio = posterior_var / (prior_var + 1e-12)
    ratio_mean = float(ratio.mean())
    frac_near1 = float(((ratio > 0.8) & (ratio < 1.2)).mean())

    print("\n【后验 vs 先验预测方差】")
    print(f"  posterior var mean/std: {posterior_var.mean():.6f} / {posterior_var.std():.6f}")
    print(f"  prior_pred var mean/std: {prior_var.mean():.6f} / {prior_var.std():.6f}")
    print(f"  ratio posterior/prior_pred: mean={ratio_mean:.3f}, frac≈1={frac_near1:.3f}")

    print("\n【诊断结论】")
    if frac_near1 > 0.5:
        print("  ⚠️  posterior 方差接近 prior 预测方差 (ratio≈1)，encoder 信息弱，存在塌缩风险")
    elif ratio_mean < 0.7:
        print(f"  ✓  posterior 方差显著小于 prior 预测方差 (mean ratio={ratio_mean:.2f})，encoder 正常")
    else:
        print(f"  📊 mean ratio={ratio_mean:.2f}，未见明显塌缩，但信息量可能偏弱")


def main():
    args = load_config()
    device = resolve_device(args.device)

    print("=" * 60)
    print("后验塌缩诊断工具 (Posterior Collapse Diagnosis)")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型: {args.ckpt}")

    # 加载数据
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

    # 获取输入维度
    batch0 = next(iter(loaders.val))
    H = batch0[0].shape[2]

    # 加载模型
    model = load_model_from_ckpt(args.ckpt, H, device)
    print(f"潜在维度: {model.latent_dim}")

    # 运行诊断
    kl_mean = diagnose_kl(model, loaders.val, device)
    ratio_random, ratio_shuffle = diagnose_intervention(model, loaders.val, device)
    diagnose_posterior_variance(model, loaders.val, device)

    # 总结
    print("\n" + "=" * 60)
    print("【总体诊断总结】")
    print("=" * 60)

    issues = []
    if kl_mean < model.latent_dim * 0.1:
        issues.append("KL 值过低")
    if ratio_random < 1.5:
        issues.append("z 对重构影响小")

    if issues:
        print(f"  ⚠️  检测到潜在问题: {', '.join(issues)}")
        print("\n  建议措施:")
        print("    1. 降低 beta 值 (当前可能过高)")
        print("    2. 减小解码器容量 (dec_hidden)")
        print("    3. 增加潜在维度 (latent_dim)")
        print("    4. 使用更激进的 KL warmup")
    else:
        print("  ✓  模型状态良好，未检测到明显的后验塌缩问题")


if __name__ == '__main__':
    main()
