#!/usr/bin/env python3
"""根据已保存的计算结果生成论文图件。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmc_model.data import find_attachment2, load_attachment2, split_topologies  # noqa: E402
from mmc_model.rsm import CubicRSM  # noqa: E402

OUT = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"R": "#4E79A7", "P": "#D9824E", "U": "#8172B2"}
NEUTRAL = "#4C5663"
ACCENT = "#B3212D"
PREFERENCE = "#E4A13A"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7.5,
            "hatch.linewidth": 0.75,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def label_panel(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(
        OUT / f"{stem}.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)


def load_model() -> tuple[pd.DataFrame, pd.DataFrame, CubicRSM]:
    """读取附件 2 并拟合三次响应面。"""
    workbook = find_attachment2(ROOT)
    frame = load_attachment2(workbook)
    no_pin, pin = split_topologies(frame)
    model = CubicRSM.fit(pin.beta, pin.eta, pin.N, pin[["R", "P", "U"]])
    return frame, no_pin, model


def plot_q1_mechanism_validation(frame: pd.DataFrame, no_pin: pd.DataFrame) -> None:
    effects = pd.read_csv(TABLES / "q1_effect_shares.csv")
    x = no_pin.eta.to_numpy() ** -2
    coef = np.linalg.lstsq(np.column_stack((np.ones(4), x)), no_pin.P, rcond=None)[0]
    pressure_dark = "#B55D34"

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), gridspec_kw={"wspace": 0.40})
    ax = axes[0]
    xx = np.linspace(x.min() * 0.98, x.max() * 1.02, 100)
    ax.plot(xx, coef[0] + coef[1] * xx, color=pressure_dark, lw=1.55)
    ax.scatter(x, no_pin.P, s=27, facecolor="white", edgecolor=COLORS["P"], lw=1.1, zorder=3)
    for eta, xi, yi in zip(no_pin.eta, x, no_pin.P):
        if np.isclose(eta, 3.0):
            offset, align = (-20, -12), "right"
        else:
            offset, align = (12, 6), "left"
        ax.annotate(
            f"$\\eta={eta:g}$",
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            ha=align,
        )
    ax.set(xlabel=r"$\eta^{-2}$", ylabel=r"无量纲压降 $P^*$")
    y_pad = 0.035 * np.ptp(no_pin.P)
    ax.set_ylim(no_pin.P.min() - y_pad, no_pin.P.max() + 2.0 * y_pad)
    ax.text(0.04, 0.94, "R² = 0.99995", transform=ax.transAxes, va="top", fontsize=6.7)
    label_panel(ax, "a")

    ax = axes[1]
    order = ["beta", "eta", "N"]
    labels = [r"$\beta$", r"$\eta$", r"$N$"]
    xpos = np.arange(3)
    width = 0.23
    for j, response in enumerate(("R", "P", "U")):
        vals = [
            effects.loc[
                (effects["response"] == response) & (effects["effect"] == factor), "share_percent"
            ].iloc[0]
            for factor in order
        ]
        ax.bar(
            xpos + (j - 1) * width,
            vals,
            width,
            color=COLORS[response],
            edgecolor="white",
            linewidth=0.55,
            label=rf"${response}^*$",
        )
    ax.set_xticks(xpos, labels)
    ax.set(ylabel="主效应平方和占比（%）", ylim=(0, 55))
    ax.grid(axis="y", color="#EEF0F2", linewidth=0.38)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.00),
        columnspacing=0.75,
        handlelength=1.0,
        handletextpad=0.35,
        fontsize=7.4,
    )
    label_panel(ax, "b")

    ax = axes[2]
    theta = frame.N * frame.beta**2
    corrected = frame.P - 0.74439372 * frame.eta**-2
    pin_mask = ~frame.beta.eq(0)
    ax.scatter(
        theta[pin_mask],
        corrected[pin_mask],
        color=COLORS["P"],
        s=12,
        alpha=0.74,
        edgecolor="none",
        zorder=2,
    )
    ax.scatter(
        theta[~pin_mask],
        corrected[~pin_mask],
        facecolor="white",
        edgecolor=NEUTRAL,
        linewidth=0.9,
        s=25,
        zorder=4,
    )
    xx = np.linspace(0, theta.max(), 100)
    ax.plot(xx, 0.03840651 + 0.09076515 * xx, color=pressure_dark, lw=1.12, zorder=1)
    ax.set(xlabel=r"针肋组合量 $N\beta^2$", ylabel=r"$P^*-0.74439\eta^{-2}$")
    ax.set_ylabel(r"$P^*-0.74439\eta^{-2}$", fontsize=7.3)
    y_span = np.ptp(corrected)
    ax.set_ylim(corrected.min() - 0.12 * y_span, corrected.max() + 0.065 * y_span)
    baseline_x = float(theta[~pin_mask].iloc[0])
    baseline_y = float(corrected[~pin_mask].iloc[0])
    ax.annotate(
        "无针肋基准",
        xy=(baseline_x, baseline_y),
        xytext=(0.12, corrected.min() - 0.055 * y_span),
        textcoords="data",
        fontsize=6.5,
        color=NEUTRAL,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.55),
    )
    label_panel(ax, "c")
    save(fig, "fig_q1_mechanism_validation")


def plot_q2_surrogate_results(model: CubicRSM) -> None:
    validation = pd.read_csv(TABLES / "q2_model_validation.csv")
    model_labels = {
        "complete polynomial d=2": "二次多项式",
        "complete polynomial d=3": "三次多项式",
        "Gaussian process Matern-5/2": "Matérn 高斯过程",
        "Gaussian process RBF": "RBF 高斯过程",
    }
    unknown_models = set(validation["model"]) - set(model_labels)
    if unknown_models:
        raise ValueError(f"发现未定义的验证模型：{sorted(unknown_models)}")
    fitted = pd.read_csv(TABLES / "q2_fitted_values.csv")
    fig = plt.figure(figsize=(7.05, 4.35))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.02, 1.0], hspace=0.62, wspace=0.42)

    ax = fig.add_subplot(grid[0, :])
    y = np.arange(len(validation))
    offsets = [-0.11, 0, 0.11]
    markers = ["o", "s", "^"]
    for response, offset, marker in zip(("R", "P", "U"), offsets, markers):
        ax.scatter(
            validation[response],
            y + offset,
            s=25,
            marker=marker,
            color=COLORS[response],
            label=rf"${response}^*$",
        )
    if np.any(validation[["R", "P", "U"]].to_numpy() <= 0):
        raise ValueError("对数坐标轴上的 RMSE 必须严格为正")
    ax.set_xscale("log")
    ax.set_yticks(y, validation["model"].map(model_labels), fontsize=8.0)
    ax.invert_yaxis()
    ax.set(xlabel="五折交叉验证 RMSE")
    ax.grid(axis="x", which="major", color="#D7DADF", lw=0.45)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.78, 1.10),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    ax.text(
        -0.035,
        1.08,
        "a",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    for j, response in enumerate(("R", "P", "U")):
        ax = fig.add_subplot(grid[1, j])
        obs = fitted[response]
        pred = fitted[f"{response}_cv_pred"]
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        span = hi - lo
        pad = 0.035 * span
        limits = (lo - pad, hi + pad)
        ax.plot(limits, limits, color="#AAB0B7", lw=0.78, ls=(0, (3, 2)), zorder=0)
        ax.scatter(obs, pred, s=11, color=COLORS[response], alpha=0.76, edgecolor="none", zorder=2)
        ax.set(xlim=limits, ylim=limits, title=rf"${response}^*$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"${response}^*$", fontsize=8.5, pad=5)
        label_panel(ax, chr(ord("b") + j))

    fig.text(0.52, 0.025, "附件值", ha="center", va="center", fontsize=8)
    fig.text(
        0.047,
        0.255,
        "三次响应面预测值",
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=8,
    )

    save(fig, "fig_q2_surrogate_validation")

    fig, axes = plt.subplots(1, 3, figsize=(7.60, 2.82), gridspec_kw={"wspace": 0.43})
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.07, right=0.985)
    fig.suptitle(r"$N=6$", y=0.985, fontsize=8.6)
    beta = np.linspace(0.10, 0.30, 160)
    eta = np.linspace(3.0, 4.5, 160)
    bb, ee = np.meshgrid(beta, eta)
    pred = model.predict(bb, ee, np.full_like(bb, 6))
    maps = [
        LinearSegmentedColormap.from_list("r", ["#EDF7F6", COLORS["R"]]),
        LinearSegmentedColormap.from_list("p", ["#FBF0EA", COLORS["P"]]),
        LinearSegmentedColormap.from_list("u", ["#F1EEF8", COLORS["U"]]),
    ]
    for j, response in enumerate(("R", "P", "U")):
        ax = axes[j]
        levels = np.linspace(pred[..., j].min(), pred[..., j].max(), 12)
        contour = ax.contourf(bb, ee, pred[..., j], levels=levels, cmap=maps[j])
        ax.contour(
            bb, ee, pred[..., j], levels=levels[::2], colors="white", linewidths=0.4, alpha=0.78
        )
        ax.set(xlabel=r"$\beta$", ylabel=r"$\eta$" if j == 0 else "", title=rf"${response}^*$")
        if j > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        cbar = fig.colorbar(
            contour, ax=ax, orientation="vertical", fraction=0.038, pad=0.025, aspect=34
        )
        cbar.ax.tick_params(labelsize=6.8, width=0.4, length=1.5, pad=1)
        cbar.set_ticks([levels[0], levels[-1]])
        label_panel(ax, chr(ord("a") + j))
    save(fig, "fig_q2_response_surfaces")


def plot_q3_multiobjective_results(results: dict) -> None:
    pareto = pd.read_csv(TABLES / "q3_pareto_grid.csv")
    ideal = np.asarray(results["q3"]["ideal"])
    nadir = np.asarray(results["q3"]["pareto_reference_upper"])
    q3 = results["q3"]["chebyshev"]
    euclidean = results["q3"]["euclidean"]

    fig, axes = plt.subplots(
        1, 3, figsize=(7.05, 2.70), gridspec_kw={"width_ratios": [1.35, 1.10, 1.0], "wspace": 0.74}
    )
    fig.subplots_adjust(top=0.78)
    ax = axes[0]
    sc = ax.scatter(
        pareto.P, pareto.R, c=pareto.U, s=5, cmap="viridis_r", alpha=0.55, rasterized=True
    )
    ax.scatter(
        q3["objectives"][1],
        q3["objectives"][0],
        marker="*",
        s=95,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax.scatter(
        euclidean["objectives"][1],
        euclidean["objectives"][0],
        marker="D",
        s=36,
        color=PREFERENCE,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax.set(xlabel=r"无量纲压降 $P^*$", ylabel=r"无量纲热阻 $R^*$")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.020, aspect=34)
    cbar.set_label(r"温度非均匀性 $U^*$", fontsize=7.3, labelpad=3)
    cbar.ax.tick_params(labelsize=6.7, pad=1)
    label_panel(ax, "a")

    ax = axes[1]
    q3z = np.asarray(q3["normalized"])
    euclidean_z = np.asarray(euclidean["normalized"])
    xx = np.arange(3)
    width = 0.34
    ax.bar(xx - width / 2, q3z, width, color=ACCENT)
    ax.bar(xx + width / 2, euclidean_z, width, color=PREFERENCE)
    ax.set_xticks(xx, [r"$z_R$", r"$z_P$", r"$z_U$"])
    ax.set(ylabel="归一化偏差", ylim=(0, 0.46))
    ax.yaxis.labelpad = 5
    label_panel(ax, "b")

    ax = axes[2]
    # 对每个 N，计算制造参数网格中 Pareto 点能够达到的最小 Chebyshev 值。
    n_values, n_scores = [], []
    for n_rows, group in pareto.groupby("N"):
        vals = group[["R", "P", "U"]].to_numpy()
        z = (vals - ideal) / (nadir - ideal)
        n_values.append(int(n_rows))
        n_scores.append(float(np.min(np.max(z, axis=1))))
    ax.plot(n_values, n_scores, color="#8F969E", lw=0.95, zorder=1)
    ax.scatter(
        n_values, n_scores, color=COLORS["R"], s=24, edgecolor="white", linewidth=0.45, zorder=2
    )
    selected_n = int(q3["design"]["N"])
    selected_score = n_scores[n_values.index(selected_n)]
    ax.scatter(
        selected_n,
        selected_score,
        marker="*",
        s=92,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.set(xlabel=r"针肋排数 $N$", ylabel=r"最小 $D_\infty$", xticks=range(2, 11))
    ax.annotate(
        rf"$N={selected_n}$" + "\n" + rf"$D_\infty={selected_score:.3f}$",
        xy=(selected_n, selected_score),
        xytext=(selected_n + 1.45, selected_score + 0.075),
        color=ACCENT,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.65),
    )
    label_panel(ax, "c")
    handles = [
        Line2D(
            [],
            [],
            marker="*",
            ls="none",
            markersize=8.5,
            markerfacecolor=ACCENT,
            markeredgecolor="white",
            label="最大偏差最小方案",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            markersize=5.2,
            markerfacecolor=PREFERENCE,
            markeredgecolor="white",
            label="欧氏距离方案",
        ),
    ]
    axes[0].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.20,
        handletextpad=0.40,
        fontsize=6.6,
    )
    save(fig, "fig_q3_pareto_compromise")


def ternary_xy(w_r, w_p, w_u):
    return np.asarray(w_p) + 0.5 * np.asarray(w_u), np.sqrt(3) / 2 * np.asarray(w_u)


def plot_q4_weight_acceptability(results: dict) -> None:
    polygons = pd.read_csv(TABLES / "q4_weight_polygons.csv")
    pareto = pd.read_csv(TABLES / "q4_weight_acceptability.csv")
    positive = pareto.query("acceptability > 0").copy()
    top = positive.nlargest(10, "acceptability")
    n_colors = mpl.colormaps["BuPu"](np.linspace(0.25, 0.88, 9))
    n_palette = {n: color for n, color in zip(range(2, 11), n_colors)}

    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 3.15), gridspec_kw={"width_ratios": [1.24, 1], "wspace": 0.56}
    )
    ax = axes[0]
    for candidate, group in polygons.groupby("candidate"):
        candidate = int(candidate)
        n_rows = int(pareto.loc[candidate, "N"])
        x, y = ternary_xy(group.w_R, group.w_P, group.w_U)
        patch = Polygon(
            np.column_stack((x, y)),
            closed=True,
            facecolor=n_palette[n_rows],
            edgecolor=(1, 1, 1, 0.30),
            linewidth=0.12,
            antialiased=True,
        )
        ax.add_patch(patch)
    triangle_x, triangle_y = ternary_xy([1, 0, 0], [0, 1, 0], [0, 0, 1])
    ax.add_patch(
        Polygon(
            np.column_stack((triangle_x, triangle_y)),
            closed=True,
            fill=False,
            edgecolor="#333333",
            linewidth=0.8,
        )
    )
    robust_idx = int(pareto.acceptability.idxmax())
    robust_poly = polygons.query("candidate == @robust_idx")
    rx, ry = ternary_xy(robust_poly.w_R, robust_poly.w_P, robust_poly.w_U)
    robust_center = (float(np.mean(rx)), float(np.mean(ry)))
    ax.add_patch(
        Polygon(
            np.column_stack((rx, ry)),
            closed=True,
            facecolor="none",
            edgecolor=ACCENT,
            linewidth=1.45,
            zorder=8,
        )
    )
    robust_area = 100 * float(pareto.loc[robust_idx, "acceptability"])
    ax.annotate(
        f"$N=4$ 代表设计\n单点 {robust_area:.2f}%",
        xy=robust_center,
        xytext=(-0.10, 0.73),
        ha="center",
        va="center",
        color=ACCENT,
        fontsize=6.8,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.70, shrinkA=2, shrinkB=2),
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=(1.0, 1.0, 1.0, 0.94),
            edgecolor=ACCENT,
            linewidth=0.55,
        ),
    )
    ax.text(-0.03, -0.04, r"$w_R$", ha="right", va="top")
    ax.text(1.03, -0.04, r"$w_P$", ha="left", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.035, r"$w_U$", ha="center", va="bottom")
    ax.set(xlim=(-0.17, 1.08), ylim=(-0.07, 0.94), aspect="equal")
    ax.axis("off")
    cmap = ListedColormap(n_colors)
    norm = BoundaryNorm(np.arange(1.5, 11.5, 1.0), cmap.N)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        ticks=np.arange(2, 11),
        boundaries=np.arange(1.5, 11.5, 1.0),
        fraction=0.055,
        pad=0.06,
        aspect=24,
    )
    cbar.set_label(r"最优整数排数 $N$", labelpad=1)
    cbar.ax.tick_params(labelsize=7.2, pad=1)
    label_panel(ax, "a")

    ax = axes[1]
    top = top.sort_values("acceptability")
    labels = [rf"$({row.beta:.3f},{row.eta:.2f},{int(row.N)})$" for row in top.itertuples()]
    values = 100 * top.acceptability.to_numpy()
    bar_colors = ["#AAA6C2"] * len(top)
    bar_colors[-1] = ACCENT
    bars = ax.barh(np.arange(len(top)), values, color=bar_colors, edgecolor="white", linewidth=0.45)
    ax.set_yticks(np.arange(len(top)), labels, fontsize=7.2)
    ax.set(xlabel="成为最优时所占权重范围（%）", ylabel=r"设计 $(\beta,\eta,N)$", xlim=(0, 5.85))
    ax.grid(axis="x", color="#E0E2E5", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            ha="left",
            va="center",
            fontsize=7.2,
            color=NEUTRAL,
        )
    label_panel(ax, "b")
    save(fig, "fig_q4_weight_acceptability")


def plot_q5_parameter_errors(results: dict) -> None:
    q5 = results["q5"]
    labels = ["Q3 nominal", "Q4 preference-robust"]
    short = ["问题三方案", "问题四代表方案"]
    fig = plt.figure(figsize=(7.05, 3.85))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=0.56, wspace=0.38)
    ax = fig.add_subplot(grid[0, :])
    width = 0.34
    positions, ticklabels = [], []
    for i, (response, color) in enumerate(COLORS.items()):
        positions.extend([2 * i, 2 * i + 0.72])
        ticklabels.extend(
            [r"$\beta$" + "\n" + rf"${response}^*$", r"$\eta$" + "\n" + rf"${response}^*$"]
        )
        for k, label in enumerate(labels):
            vals = np.asarray(q5[label]["local_sensitivity_rows_R_P_U_cols_beta_eta"])[i]
            bars = ax.bar(
                np.array([2 * i, 2 * i + 0.72]) + (k - 0.5) * width,
                vals,
                width,
                color=color if k == 0 else "white",
                hatch=None if k == 0 else "////",
                edgecolor=color,
                linewidth=0.95,
            )
            for bar, value in zip(bars, vals):
                if value < 0.03:
                    shown = "<0.001" if value < 0.001 else f"{value:.3f}"
                    x_shift = -4 if k == 0 else 4
                    ax.annotate(
                        shown,
                        xy=(bar.get_x() + bar.get_width() / 2, max(value, 0.002)),
                        xytext=(x_shift, 9),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.45),
                    )
    ax.set_xticks(positions, ticklabels)
    ax.set(ylabel="相对敏感度")
    ax.axvline(1.36, color="#EEF0F2", lw=0.48)
    ax.axvline(4.08, color="#EEF0F2", lw=0.48)
    label_panel(ax, "a")

    worst = pd.read_csv(TABLES / "q5_worst_case.csv")
    for j, response in enumerate(("R", "P", "U")):
        ax = fig.add_subplot(grid[1, j])
        for k, label in enumerate(labels):
            part = worst.query("design == @label and response == @response")
            ax.plot(
                part.radius_percent,
                part.degradation_percent,
                marker="o",
                ms=3.5,
                markerfacecolor=COLORS[response] if k == 0 else "white",
                markeredgecolor=COLORS[response],
                markeredgewidth=0.8,
                color=COLORS[response],
                lw=1.25,
                alpha=1.0,
                ls="-" if k == 0 else "--",
                label=short[k],
            )
        ax.set(
            xlabel="参数误差（%）",
            ylabel="最大增幅（%）",
            title=rf"${response}^*$",
            xticks=[1, 3, 5],
        )
        label_panel(ax, chr(ord("b") + j))
    handles = [
        Line2D(
            [],
            [],
            color=NEUTRAL,
            lw=1.25,
            marker="o",
            markersize=4.0,
            markerfacecolor=NEUTRAL,
            label=short[0],
        ),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            lw=1.25,
            ls="--",
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=NEUTRAL,
            label=short[1],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    save(fig, "fig_q5_parameter_error_analysis")


def main() -> None:
    set_style()
    frame, no_pin, model = load_model()
    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    plot_q1_mechanism_validation(frame, no_pin)
    plot_q2_surrogate_results(model)
    plot_q3_multiobjective_results(results)
    plot_q4_weight_acceptability(results)
    plot_q5_parameter_errors(results)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
