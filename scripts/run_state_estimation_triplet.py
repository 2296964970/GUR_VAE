import argparse
import os
import subprocess
import sys
from pathlib import Path


def to_matlab_path(p: str) -> str:
    return str(Path(p).resolve()).replace('\\', '/')

def main() -> None:
    p = argparse.ArgumentParser(description='Run MATLAB WLS state estimation from triplet wide CSVs (attacked/repaired/clean).')
    p.add_argument('--matlab_root', type=str, required=True, help='Path to MATLAB repo root (loadseries)')
    p.add_argument('--matlab_bin', type=str, default='matlab', help='MATLAB executable (e.g., matlab, "C:/Program Files/MATLAB/R2022b/bin/matlab.exe")')
    p.add_argument('--case', type=str, default='case14')
    p.add_argument('--attacked_csv', type=str, required=True, help='Path to tail_attacked_wide.csv')
    p.add_argument('--repaired_csv', type=str, required=True, help='Path to tail_repaired_wide.csv')
    p.add_argument('--clean_csv', type=str, required=True, help='Path to tail_true_wide.csv (clean baseline)')
    p.add_argument('--out_dir', type=str, default='', help='Directory to write SE figures/metrics (default: output/<case>/se_eval)')
    p.add_argument('--format', type=str, default='pdf', choices=['pdf', 'eps'])
    p.add_argument('--angle_unit', type=str, default='rad', choices=['rad', 'deg'])
    p.add_argument('--reference_mode', type=str, default='eliminate', choices=['eliminate', 'pseudo', 'none'])
    args = p.parse_args()

    # Validate inputs
    for k in ('matlab_root', 'attacked_csv', 'repaired_csv', 'clean_csv'):
        v = getattr(args, k)
        if not os.path.exists(v):
            print(f'[error] Missing path for --{k}: {v}', file=sys.stderr)
            sys.exit(2)

    out_dir = args.out_dir or os.path.join('output', args.case, 'se_eval')
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join('runs', 'matlab_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare MATLAB script (self-contained) that reads triplet CSVs and runs WLS
    mfile = os.path.join(tmp_dir, 'run_se_triplet_tmp.m')
    code_lines = []
    code_lines.append(f"addpath(genpath('{to_matlab_path(args.matlab_root)}')); try, init; catch, end")
    code_lines.append(f"case_name = '{args.case}';")
    code_lines.append(f"attacked_csv = '{to_matlab_path(args.attacked_csv)}';")
    code_lines.append(f"repaired_csv = '{to_matlab_path(args.repaired_csv)}';")
    code_lines.append(f"clean_csv    = '{to_matlab_path(args.clean_csv)}';")
    code_lines.append(f"out_dir      = '{to_matlab_path(out_dir)}';")
    code_lines.append(f"fmt          = '{args.format}';")
    code_lines.append(f"angle_unit   = '{args.angle_unit}';")
    code_lines.append(f"reference_mode = '{args.reference_mode}'; %#ok<NASGU>")
    code_lines.append("if ~exist(out_dir,'dir'), mkdir(out_dir); end; fig_dir = fullfile(out_dir,'figures'); metrics_dir = fullfile(out_dir,'metrics'); if ~exist(fig_dir,'dir'), mkdir(fig_dir); end; if ~exist(metrics_dir,'dir'), mkdir(metrics_dir); end")
    code_lines.append("Ta = readtable(attacked_csv, 'TextType','string'); Tr = readtable(repaired_csv, 'TextType','string'); Tc = readtable(clean_csv, 'TextType','string');")
    code_lines.append("tsa = string(Ta{:,1}); tsr = string(Tr{:,1}); tsc = string(Tc{:,1});")
    code_lines.append("ts_common = intersect(intersect(tsa, tsr), tsc);")
    code_lines.append("if isempty(ts_common), error('No common timestamps across triplet CSVs.'); end")
    code_lines.append("[~, order] = ismember(ts_common, tsr); [~, ix] = sort(order); ts_vals = ts_common(ix);")
    code_lines.append("[~, ~, segments0, ~, Y] = estimation.parse_row_robust(case_name, clean_csv, ts_vals(1)); nb = segments0.nb;")
    code_lines.append("n = numel(ts_vals); R = table('Size',[n 1], 'VariableTypes', {'string'}, 'VariableNames', {'timestamp'}); R.timestamp = ts_vals(:);")
    code_lines.append("vars = { 'converged_att','iter_att','cost_att','maxnres_att','meannres_att','vmrmse_att','varmse_att', 'converged_rep','iter_rep','cost_rep','maxnres_rep','meannres_rep','vmrmse_rep','varmse_rep', 'converged_cln','iter_cln','cost_cln','maxnres_cln','meannres_cln' };")
    code_lines.append("for k=1:numel(vars), R.(vars{k}) = NaN(n,1); end")
    code_lines.append("for k=1:n")
    code_lines.append("  ts = ts_vals(k); ts_tag = char(util.sanitize_timestamp(ts));")
    code_lines.append("  try")
    code_lines.append("    [zC, ~, segC, ~, ~] = estimation.parse_row_robust(case_name, clean_csv, ts);")
    code_lines.append("    modelC = estimation.build_model_rect(case_name, segC, zC, Y);")
    code_lines.append("    wC = estimation.build_weights_rect(segC, zC, [], 'UseBlocks', true);")
    code_lines.append("    [stC, infoC] = estimation.wls_rect(modelC, wC, struct('MaxIter',20,'TolX',1e-8,'TolFun',1e-8), struct());")
    code_lines.append("    VmC = hypot(stC.Vr, stC.Vi); VaC = util.wrapToPi(atan2(stC.Vi, stC.Vr));")
    code_lines.append("    R.converged_cln(k) = double(infoC.converged); R.iter_cln(k) = infoC.iter; R.cost_cln(k) = full(infoC.cost); R.maxnres_cln(k) = full(infoC.max_norm_residual); R.meannres_cln(k) = full(infoC.mean_norm_residual);")
    code_lines.append("  catch ME")
    code_lines.append("    warning('Clean failed @ %s: %s', ts, ME.message); VmC = nan(nb,1); VaC = nan(nb,1);")
    code_lines.append("  end")
    code_lines.append("  try")
    code_lines.append("    [zA, ~, segA, ~, ~] = estimation.parse_row_robust(case_name, attacked_csv, ts);")
    code_lines.append("    modelA = estimation.build_model_rect(case_name, segA, zA, Y);")
    code_lines.append("    wA = estimation.build_weights_rect(segA, zA, [], 'UseBlocks', true);")
    code_lines.append("    [stA, infoA] = estimation.wls_rect(modelA, wA, struct('MaxIter',20,'TolX',1e-8,'TolFun',1e-8), struct());")
    code_lines.append("    VmA = hypot(stA.Vr, stA.Vi); VaA = util.wrapToPi(atan2(stA.Vi, stA.Vr));")
    code_lines.append("    R.converged_att(k) = double(infoA.converged); R.iter_att(k) = infoA.iter; R.cost_att(k) = full(infoA.cost); R.maxnres_att(k) = full(infoA.max_norm_residual); R.meannres_att(k) = full(infoA.mean_norm_residual);")
    code_lines.append("  catch ME")
    code_lines.append("    warning('Attacked failed @ %s: %s', ts, ME.message); VmA = nan(nb,1); VaA = nan(nb,1);")
    code_lines.append("  end")
    code_lines.append("  try")
    code_lines.append("    [zR, ~, segR, ~, ~] = estimation.parse_row_robust(case_name, repaired_csv, ts);")
    code_lines.append("    modelR = estimation.build_model_rect(case_name, segR, zR, Y);")
    code_lines.append("    wR = estimation.build_weights_rect(segR, zR, [], 'UseBlocks', true);")
    code_lines.append("    [stR, infoR] = estimation.wls_rect(modelR, wR, struct('MaxIter',20,'TolX',1e-8,'TolFun',1e-8), struct());")
    code_lines.append("    VmR = hypot(stR.Vr, stR.Vi); VaR = util.wrapToPi(atan2(stR.Vi, stR.Vr));")
    code_lines.append("    R.converged_rep(k) = double(infoR.converged); R.iter_rep(k) = infoR.iter; R.cost_rep(k) = full(infoR.cost); R.maxnres_rep(k) = full(infoR.max_norm_residual); R.meannres_rep(k) = full(infoR.mean_norm_residual);")
    code_lines.append("  catch ME")
    code_lines.append("    warning('Repaired failed @ %s: %s', ts, ME.message); VmR = nan(nb,1); VaR = nan(nb,1);")
    code_lines.append("  end")
    code_lines.append("  R.vmrmse_att(k) = sqrt(nanmean((VmA - VmC).^2)); R.vmrmse_rep(k) = sqrt(nanmean((VmR - VmC).^2));")
    code_lines.append("  dVa_att = util.wrapToPi(VaA - VaC); dVa_rep = util.wrapToPi(VaR - VaC);")
    code_lines.append("  R.varmse_att(k) = sqrt(nanmean(dVa_att.^2)); R.varmse_rep(k) = sqrt(nanmean(dVa_rep.^2));")
    code_lines.append("  try")
    code_lines.append("    vis='off';")
    code_lines.append("    f_vm = figure('Name', sprintf('%s |V| @ %s', upper(char(case_name)), char(ts)), 'Color','w','Visible',vis);")
    code_lines.append("    plot(1:nb, VmA,'-o','LineWidth',1.2); hold on; grid on; plot(1:nb, VmR,'--s','LineWidth',1.2); plot(1:nb, VmC,':x','LineWidth',1.2); legend({'Attacked','Repaired','Clean'},'Location','best'); xlabel('Bus'); ylabel('|V| (p.u.)');")
    code_lines.append("    exportgraphics(f_vm, fullfile(fig_dir, sprintf('%s_vm_%s.%s', upper(char(case_name)), util.sanitize_timestamp(ts), lower(fmt))), 'ContentType','vector'); try, close(f_vm); catch, end")
    code_lines.append("    if strcmpi(angle_unit,'deg'), VaAplt=rad2deg(VaA); VaRplt=rad2deg(VaR); VaCplt=rad2deg(VaC); ylab='Angle (deg)'; ylims=[-180 180]; else, VaAplt=VaA; VaRplt=VaR; VaCplt=VaC; ylab='Angle (rad)'; ylims=[-pi pi]; end")
    code_lines.append("    f_va = figure('Name', sprintf('%s angle @ %s', upper(char(case_name)), char(ts)), 'Color','w','Visible',vis);")
    code_lines.append("    plot(1:nb, VaAplt,'-o','LineWidth',1.2); hold on; grid on; plot(1:nb, VaRplt,'--s','LineWidth',1.2); plot(1:nb, VaCplt,':x','LineWidth',1.2); legend({'Attacked','Repaired','Clean'},'Location','best'); xlabel('Bus'); ylabel(ylab); ylim(ylims);")
    code_lines.append("    exportgraphics(f_va, fullfile(fig_dir, sprintf('%s_va_%s.%s', upper(char(case_name)), util.sanitize_timestamp(ts), lower(fmt))), 'ContentType','vector'); try, close(f_va); catch, end")
    code_lines.append("  catch")
    code_lines.append("  end")
    code_lines.append("end")
    code_lines.append("out_csv = fullfile(metrics_dir, sprintf('%s_state_estimation_metrics_triplet.csv', char(case_name))); writetable(R, out_csv);")
    code_lines.append("fprintf('[se_triplet] Metrics -> %s\\n', out_csv);")
    code = "\n".join(code_lines)

    with open(mfile, 'w', encoding='utf-8') as f:
        f.write(code)

    # Run MATLAB in batch
    matlab_cmd = [args.matlab_bin, '-batch', f"run('{to_matlab_path(mfile)}')"]
    print('[info] Launching MATLAB batch:', ' '.join(matlab_cmd))
    proc = subprocess.run(matlab_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    print(proc.stdout)
    print('[ok] State estimation finished. Artifacts at:', out_dir)


if __name__ == '__main__':
    main()
