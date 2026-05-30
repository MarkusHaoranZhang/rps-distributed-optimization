"""管线级冒烟：只跑 figure 1（最小代价）确认 main.py 入口能 end-to-end 跑通。

这个测试需要写文件，所以放在 tests/ 而不是单元测试。CI 跑它能确保 figure
绘制函数和 RPSConfig 默认值在新环境下都还工作。
"""

import os

import numpy as np
import pytest

from config import RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph
from experiments import run_optimization
from figures import plot_figure1


def test_figure1_pipeline_endtoend(tmp_path):
    """跑一个最小 N=8 / T=200 的 RPS-Full 实验，画出 figure 1 风格的图。"""
    N, d, T = 8, 4, 200
    W, adj, _ = build_graph(N, seed=0)
    A_list, b_list = generate_least_squares_data(N, d, 3, seed=0)
    cost = LeastSquaresCost(A_list, b_list)
    cfg = RPSConfig(burn_in=80, window_len=20, top_m=8, diagnose_every=5)
    fault_cfg = {'onset': 100, 'agents': [0], 'type': 'constant',
                 'delta': 0.01 * np.ones(d)}

    err, residuals, log = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg, seed=0,
    )
    assert err.shape == (T,)
    assert residuals.shape == (T, N)
    assert "gamma_history" in log

    # 实际写一张图
    out_pdf = tmp_path / "smoke_fig.pdf"
    plot_figure1(residuals, faulty_idx=0, direct_idx=1, twohop_idx=2,
                 fault_onset=100, save_path=str(out_pdf))
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 1000   # 非空 PDF


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="CI 跑得慢，只在本地跑这个稍重的冒烟",
)
def test_main_py_quick_figures_1(tmp_path, monkeypatch):
    """验证 ``python main.py --quick --figures 1`` 等价的程序化调用能跑通。"""
    import argparse

    monkeypatch.chdir(tmp_path)
    import main as main_mod

    monkeypatch.setattr(
        main_mod, "parse_args",
        lambda: argparse.Namespace(quick=True, figures="1",
                                    seed=0, mc=None,
                                    dataset="synthetic"),
    )
    main_mod.main()
    assert (tmp_path / "fig_preliminary.pdf").exists()
