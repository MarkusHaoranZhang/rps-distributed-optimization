"""
Plotting scripts: produce the eight figures from the paper
===========================================================

Every figure consumes data produced by real experiments; no distribution
parameter is hard-coded.

We use matplotlib's non-interactive ``Agg`` backend so each ``plot_figureN``
saves a PDF directly without opening a GUI window. This keeps figure
generation stable in headless CI / remote environments.
"""

from __future__ import annotations

import matplotlib

# The Agg backend must be selected **before** ``pyplot`` is imported. If
# pyplot has already been imported (e.g. inside Jupyter), ``use("Agg")``
# is silently ignored and figure generation breaks in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  must come after ``use("Agg")``
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
    # Residual norms span several orders of magnitude (10^-4 to 10^0); a
    # log y-axis keeps spatial attenuation visible. On a linear y-axis
    # the two-hop residual is squashed near 0 and disappears.
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
    """Four-panel parameter sensitivity.

    The y-axis uses a log scale: under a parameter sweep the final error
    can span several orders of magnitude (e.g. very small tau makes the
    RPS discount too aggressive and slows convergence; ``hop=1`` shrinks
    the diagnosable scope and the error blows up). A linear y-axis flattens
    the small-error points into invisibility.
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
    """Three-panel stress test.

    Parameters
    ----------
    deltas       : array of fault magnitudes.
    gaps         : performance gap (%, relative) of RPS-Full over
                   Hard-Threshold.
    loss_rates   : communication packet-loss rates (%).
    perf_retain  : RPS-Full's final relative error at each loss rate
                   (dimensionless, not a percentage). What the reader
                   cares about is the trend: error grows with loss. The
                   "retains 80% of advantage" phrase from Section 4.5.3
                   of the paper is not computed directly in this figure
                   (to avoid misleading 0/0 divisions); the specific
                   threshold is left to the prose.
    nfaults      : number of simultaneously faulty agents.
    accs         : RPS-Full's final relative error at each fault count.
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
    # Use set_xticks for tick labels: this stays compatible with
    # matplotlib 3.5 .. current. The ``labels=`` kwarg of boxplot was
    # deprecated in 3.9, and ``tick_labels=`` does not exist before 3.9.
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
    """Diagnostic-delay distribution (paper Figure 4; ``fig 7`` in this
    codebase).

    Parameters
    ----------
    ht_mtcd  : 1-D array, Hard-Threshold's MTCD distribution over MC trials.
    rps_mtcd : 1-D array, RPS-Full's MTCD distribution over MC trials.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [np.asarray(ht_mtcd, dtype=float),
            np.asarray(rps_mtcd, dtype=float)]
    labels = ["Hard-Threshold", "RPS-Full"]
    colors = [C3, C0]
    # Same as plot_figure6: use set_xticklabels rather than boxplot's
    # labels= / tick_labels= for cross-version (3.5 .. current)
    # compatibility.
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
