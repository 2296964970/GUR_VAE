"""Collect alpha sweep summary from log files and write a compact TSV.

Usage:
  python scripts/collect_alpha_summary.py \
      --logs_dir experiment_logs/sweep_alpha \
      --out_tsv experiment_logs/sweep_alpha/summary.tsv

Assumptions:
  - Each log file name encodes alpha as suffix, e.g., alpha_0p01.log
  - Logs contain one step under the 'Summary' header and two aggregate lines.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

AGG_RE = re.compile(
    r"Aggregate \(mean across steps\):\s*"
    r"MSE_att=(?P<mse_b>[-+]?\d*\.?\d+),\s*"
    r"MSE_rep=(?P<mse>[-+]?\d*\.?\d+),\s*"
    r"RMSE_att=(?P<rmse_b>[-+]?\d*\.?\d+),\s*"
    r"RMSE_rep=(?P<rmse>[-+]?\d*\.?\d+),\s*"
    r"Gain_MSE=(?P<rim_mse>[-+]?\d*\.?\d+)%.*,\s*Gain_RMSE=(?P<rim_rmse>[-+]?\d*\.?\d+)%"
)

DROP_AGG_RE = re.compile(
    r"Drop-Only Aggregate:\s*"
    r"MSE_rep@Drop=(?P<mse_drop>[-+]?\d*\.?\d+|nan),\s*"
    r"RMSE_rep@Drop=(?P<rmse_drop>[-+]?\d*\.?\d+|nan),\s*"
    r"NRMSE_rep@Drop=(?P<nrmse_drop>[-+]?\d*\.?\d+|nan)"
)


def parse_log(path: str) -> Dict[str, float | int | str]:
    raw = None
    with open(path, 'rb') as f:
        raw = f.read()
    text = None
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    step_info: Dict[str, float | int | str] = {}
    # Prefer parsing from the info line (robust to whitespace):
    # [info] Step 0: ... observed_tail_count=144, dropped=43 (29.86%)
    for ln in lines:
        m = re.search(r"dropped=(?P<drop>\d+)\s*\((?P<dropr>[-+]?\d*\.?\d+)%\)", ln)
        if m:
            step_info.update({'dropped': int(m.group('drop')), 'drop_rate_pct': float(m.group('dropr'))})
            break
    # aggregates
    for ln in lines:
        m = AGG_RE.search(ln)
        if m:
            g = m.groupdict()
            step_info.update({
                'mse_att': float(g['mse_b']),
                'mse_rep': float(g['mse']),
                'rmse_att': float(g['rmse_b']),
                'rmse_rep': float(g['rmse']),
                'gain_mse_pct': float(g['rim_mse']),
                'gain_rmse_pct': float(g['rim_rmse']),
            })
    for ln in lines:
        m = DROP_AGG_RE.search(ln)
        if m:
            g = m.groupdict()
            def _f(x: str) -> float:
                try:
                    return float(x)
                except Exception:
                    return float('nan')
            step_info.update({
                'mse_rep_drop': _f(g['mse_drop']),
                'rmse_rep_drop': _f(g['rmse_drop']),
                'nrmse_rep_drop': _f(g['nrmse_drop']),
            })
    return step_info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--logs_dir', type=str, required=True)
    p.add_argument('--out_tsv', type=str, required=True)
    args = p.parse_args()

    entries: List[Dict[str, str | float | int]] = []
    for name in sorted(os.listdir(args.logs_dir)):
        if not name.startswith('alpha_') or not name.endswith('.log'):
            continue
        alpha = name[len('alpha_'):-len('.log')]
        info = parse_log(os.path.join(args.logs_dir, name))
        if info:
            info_row: Dict[str, str | float | int] = {'alpha': alpha}
            info_row.update(info)
            entries.append(info_row)

    cols = [
        'alpha', 'dropped', 'drop_rate_pct',
        'mse_att', 'mse_rep', 'rmse_att', 'rmse_rep',
        'gain_mse_pct', 'gain_rmse_pct',
        'mse_rep_drop', 'rmse_rep_drop', 'nrmse_rep_drop',
    ]

    with open(args.out_tsv, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for row in entries:
            vals = [row.get(k, '') for k in cols]
            f.write('\t'.join(str(v) for v in vals) + '\n')

    print(f"[ok] Wrote summary: {args.out_tsv} (rows={len(entries)})")


if __name__ == '__main__':
    main()
