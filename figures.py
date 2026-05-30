"""
出图脚本：生成论文 8 张图
==========================

每张图都接入真实实验产出的数据；不再硬编码任何分布参数。

使用 matplotlib 的非交互 ``Agg`` 后端：所有 ``plot_figureN`` 都直接保存
PDF，不弹出 GUI 窗口。这让出图在无显示的 CI / 远程环境下也能稳定运行。
"""

from __future__ import annotations

import matplotlib

# Agg 后端必须在 import pyplot 之前设置——否则在已经导入 pyplot 的环境
# （例如 Jupyter）里 use("Agg") 会被忽略，导致无 GUI 环境出图失败。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  必须在 use("Agg") 之后
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

C0 = "#1f77b4"
C1 = "#ff7f0e"
C2 = "#2ca02c"
C3 = "#d62728"
C4 = "#9467bd"
C5 = "#8c564b"
C7 = "#7f7f7f"
METHOD_COLORS = {
    "Ideal": C7,
    "Hard-Threshold": C3,
    "Uniform-Discount": C1,
    "Byzantine-Resilient": C4,
    "RPS-Symmetric": C5,
    "RPS-NoOrder": "#bcbd22",
    "RPS-Full": C0,
}
CONVERGENT_COLOR = C0
DIVERGENT_COLOR = C3
THEORY_COLOR = C2


# ---------------------------------------------------------------------------
# Figure 1 : Residual evolution under soft fault injection
# ---------------------------------------------------------------------------

def plot_figure1(residuals, faulty_idx, direct_idx, twohop_idx, fault_onset,
                 save_path="fig_preliminary.pdf"):
    t = np.arange(residuals.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    # 残差范数横跨多个数量级（10^-4 到 10^0），用 log y 让 spatial attenuation
    # 可见。线性 y 轴下两-hop 残差会被压缩到 0 附近不可见。
    ax.set_yscale('log')
    ax.plot(t, residuals[:, faulty_idx], label="Faulty agent", linewidth=1.0,
            alpha=0.8, color=C2)
    ax.plot(t, residuals[:, direct_idx], label="Direct neighbor", linewidth=1.2, color=C0)
    ax.plot(t, residuals[:, twohop_idx], label="Two-hop neighbor", linewidth=1.2, color=C1)
    ax.axvline(fault_onset, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.7, label="Fault onset")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual norm"); ax.legend()
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 : Comparative convergence trajectories
# ---------------------------------------------------------------------------

def plot_figure2(fig2_data, methods, scenarios, save_path="fig_comparative.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    iters = np.arange(len(fig2_data[scenarios[0]]["Ideal"]))
    for ax, sc in zip(axes, scenarios):
        for m in methods:
            if m == "Ideal":
                ls, lw, alpha = "--", 1.8, 0.6
            elif m == "RPS-Full":
                ls, lw, alpha = "-", 2.2, 1.0
            else:
                ls, lw, alpha = "-", 1.2, 0.55
            ax.semilogy(iters, fig2_data[sc][m], label=m, linestyle=ls,
                        linewidth=lw, alpha=alpha,
                        color=METHOD_COLORS.get(m, "k"))
        ax.set_title(sc); ax.set_xlabel("Iteration"); ax.set_ylabel("Relative error")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 : Parameter sensitivity
# ---------------------------------------------------------------------------

def plot_figure3(fig3_data, save_path="fig_sensitivity.pdf"):
    """参数敏感性 4 子图。

    y 轴用 log scale：参数扫描里 final error 容易跨多个量级（如 τ 取极小值时
    RPS 折扣过强收敛变慢；hop=1 时诊断作用域不足导致 error 飙升）。线性 y
    会把小 error 点压扁。
    """
    xlabels = ["$s$", "$\\eta$", "$\\tau$ scaling", "$h$"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, (x, y)), xl in zip(axes.flat, fig3_data.items(), xlabels):
        ax.plot(x, np.asarray(y), marker='o', linestyle='-', linewidth=1.8,
                markersize=7, color=C0, markerfacecolor=C0,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.set_yscale('log')
        ax.set_xlabel(xl)
        ax.set_ylabel("Final relative error")
        ax.set_title(name); ax.grid(True, alpha=0.3, linestyle="--",
                                       which='both')
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 : Empirical stability phase diagram
# ---------------------------------------------------------------------------

def plot_figure4(alphas, etas_inv, conv_mask, kappa_emp,
                 kappa_theo=None,
                 save_path="fig_stability.pdf"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(alphas[conv_mask], etas_inv[conv_mask], c=CONVERGENT_COLOR,
               marker='o', s=35, label='Convergent', alpha=0.6, edgecolors='none')
    ax.scatter(alphas[~conv_mask], etas_inv[~conv_mask], c=DIVERGENT_COLOR,
               marker='x', s=35, label='Divergent', alpha=0.7, linewidth=1.2)
    alp_line = np.logspace(np.log10(max(alphas.min(), 1e-4)),
                           np.log10(alphas.max()), 200)
    ax.plot(alp_line, alp_line / kappa_emp, '--', color=THEORY_COLOR,
            linewidth=2.0, label='Empirical $\\eta^{-1} = \\alpha / \\kappa_{emp}$')
    if kappa_theo is not None and kappa_theo > 0:
        ax.plot(alp_line, alp_line / kappa_theo, ':', color="#7f7f7f",
                linewidth=2.0,
                label='Theoretical $\\kappa$ (Theorem 1)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("$\\alpha$"); ax.set_ylabel("$\\eta^{-1}$")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 : Stress tests
# ---------------------------------------------------------------------------

def plot_figure5(deltas, gaps, loss_rates, perf_retain, nfaults, accs,
                 save_path="fig_stress.pdf"):
    """三子图压力测试。

    Parameters
    ----------
    deltas       : 故障幅度数组
    gaps         : RPS-Full vs Hard-Threshold 性能差（%，相对值）
    loss_rates   : 通信丢包率（%）
    perf_retain  : RPS-Full 在不同丢包率下的 final relative error（无量纲，
                   不是百分比）。读者关心的是 trend：丢包越大 error 越大；
                   论文 4.5.3 文字描述的 "retains 80% of advantage" 不在此
                   图直接计算（避免 0/0 折算误导），具体阈值留给文字描述。
    nfaults      : 同时故障 agent 数
    accs         : RPS-Full 在不同故障数下的 final relative error
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(deltas, gaps, marker='s', color=C0, linewidth=1.8, markersize=7,
                 markerfacecolor=C0, markeredgecolor='white', markeredgewidth=0.8)
    axes[0].set_xlabel("$\\Delta$")
    axes[0].set_ylabel("Performance gap over hard threshold (%)")
    axes[0].set_title("High fault magnitude")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].plot(loss_rates, perf_retain, marker='^', color=C0, linewidth=1.8,
                 markersize=7, markerfacecolor=C0, markeredgecolor='white',
                 markeredgewidth=0.8)
    axes[1].set_yscale('log')
    axes[1].set_xlabel("Packet loss rate (%)")
    axes[1].set_ylabel("Final relative error")
    axes[1].set_title("Communication degradation")
    axes[1].grid(True, alpha=0.3, linestyle="--", which='both')

    axes[2].plot(nfaults, accs, marker='o', color=C0, linewidth=1.8, markersize=7,
                 markerfacecolor=C0, markeredgecolor='white', markeredgewidth=0.8)
    axes[2].set_yscale('log')
    axes[2].set_xlabel("Number of simultaneous faulty agents")
    axes[2].set_ylabel("Final relative error")
    axes[2].set_title("Multiple simultaneous faults")
    axes[2].grid(True, alpha=0.3, linestyle="--", which='both')

    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 : Ablation effect sizes
# ---------------------------------------------------------------------------

def plot_figure6(full_err, noorder_err, sym_err, save_path="fig_ablation.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [full_err * 1e3, noorder_err * 1e3, sym_err * 1e3]
    labels = ["RPS-Full", "RPS-NoOrder", "RPS-Symmetric"]
    colors = [C0, C3, C1]
    # 用 set_xticks 设置 tick label，兼容 matplotlib 3.5 - 当前所有版本：
    # `labels=` kwarg 在 3.9 起 deprecated，`tick_labels=` 在 < 3.9 不存在。
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Final relative error ($\\times10^{-3}$)")
    ax.set_title("Ablation effect sizes")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 : Diagnostic delay (MTCD)
# ---------------------------------------------------------------------------

def plot_figure7(ht_mtcd, rps_mtcd, save_path="fig_diagnostic.pdf"):
    """诊断延迟分布对比图（论文 Figure 4 / 代码内 fig 7）。

    Parameters
    ----------
    ht_mtcd  : 1D 数组，Hard-Threshold 在 MC trials 上的 MTCD 分布
    rps_mtcd : 1D 数组，RPS-Full 在 MC trials 上的 MTCD 分布
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [np.asarray(ht_mtcd, dtype=float),
            np.asarray(rps_mtcd, dtype=float)]
    labels = ["Hard-Threshold", "RPS-Full"]
    colors = [C3, C0]
    # 同 plot_figure6：用 set_xticklabels 而非 boxplot 的 labels=/tick_labels=
    # 跨 matplotlib 版本（3.5 - 当前）兼容。
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("MTCD (iterations)")
    ax.set_title("Diagnostic delay distribution")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 : Scale invariance
# ---------------------------------------------------------------------------

def plot_figure8(curves, save_path="fig_scaling.pdf"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for N, err in curves.items():
        ax.semilogy(err, label=f"N={N}", linewidth=1.8)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Relative error")
    ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Scale invariance")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
