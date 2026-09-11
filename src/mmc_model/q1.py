"""问题一的描述性计算与机理检验。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dual_manifold_flow_balance(branch_flow: np.ndarray) -> dict[str, np.ndarray]:
    """沿成对入口、出口歧管的控制体链累积流量。

    参数
    ----
    branch_flow:
        按歧管方向排列的一维非负微通道支路流量 ``q_i``。假定全部入口流量
        均流经这些支路，随后由出口歧管汇集。

    返回值
    ------
    dict
        ``inlet`` 和 ``outlet`` 分别给出第一条支路之前及各支路之后的歧管
        流量。因此，入口序列从 ``sum(q_i)`` 递减至零，出口序列则从零递增
        至相同的总流量。
    """
    q = np.asarray(branch_flow, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("branch_flow 必须是非空的一维数组")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("branch_flow 必须只包含有限的非负数值")

    inlet = np.empty(q.size + 1, dtype=float)
    outlet = np.empty(q.size + 1, dtype=float)
    inlet[0] = q.sum()
    outlet[0] = 0.0
    for i, flow in enumerate(q):
        inlet[i + 1] = inlet[i] - flow
        outlet[i + 1] = outlet[i] + flow

    inlet[-1] = 0.0
    outlet[-1] = q.sum()
    return {"branch": q.copy(), "inlet": inlet, "outlet": outlet}


def coolant_energy_balance() -> dict[str, float]:
    """估计芯片发热功率和冷却液平均温升。"""
    chip_volume = 6e-3 * 6e-3 * 200e-6
    heat = 5e9 * chip_volume
    delta_t = heat / (1e-3 * 4182.0)
    return {
        "chip_volume_m3": chip_volume,
        "heat_W": heat,
        "mean_coolant_rise_K": delta_t,
    }


def no_pin_pressure_fit(no_pin: pd.DataFrame) -> dict[str, float]:
    """拟合无针肋压降与歧管深高比平方倒数之间的关系。"""
    x = np.column_stack((np.ones(len(no_pin)), no_pin.eta.to_numpy() ** -2))
    y = no_pin.P.to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "intercept": float(coef[0]),
        "eta_inv2": float(coef[1]),
        "r2": float(1 - ss_res / ss_tot),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def mechanism_pressure_fit(pin: pd.DataFrame) -> dict[str, float]:
    """拟合问题一中由机理分析得到的简化压降关系。"""
    x = np.column_stack(
        (np.ones(len(pin)), pin.eta.to_numpy() ** -2, pin.N.to_numpy() * pin.beta.to_numpy() ** 2)
    )
    y = pin.P.to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "intercept": float(coef[0]),
        "eta_inv2": float(coef[1]),
        "N_beta2": float(coef[2]),
        "r2": float(1 - ss_res / ss_tot),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def factorial_effects(pin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算各参数水平的均值及平衡设计中的经典平方和。"""
    response_cols = ["R", "P", "U"]
    means = []
    for factor in ("beta", "eta", "N"):
        part = pin.groupby(factor, as_index=False)[response_cols].mean()
        part.insert(0, "factor", factor)
        part = part.rename(columns={factor: "level"})
        means.append(part)
    level_means = pd.concat(means, ignore_index=True)

    total_ss = {r: float(((pin[r] - pin[r].mean()) ** 2).sum()) for r in response_cols}
    rows = []
    for factor in ("beta", "eta", "N"):
        counts = pin.groupby(factor).size()
        grouped = pin.groupby(factor)[response_cols].mean()
        for r in response_cols:
            ss = float(
                sum(
                    counts.loc[level] * (grouped.loc[level, r] - pin[r].mean()) ** 2
                    for level in grouped.index
                )
            )
            rows.append(
                {"response": r, "effect": factor, "ss": ss, "share_percent": 100 * ss / total_ss[r]}
            )
    # 在剔除 beta 和 N 各自主效应后计算二者的交互作用。
    for r in response_cols:
        grand = pin[r].mean()
        bm = pin.groupby("beta")[r].mean()
        nm = pin.groupby("N")[r].mean()
        cell = pin.groupby(["beta", "N"])[r].mean()
        reps = pin.groupby(["beta", "N"]).size()
        ss = sum(
            reps.loc[idx] * (cell.loc[idx] - bm.loc[idx[0]] - nm.loc[idx[1]] + grand) ** 2
            for idx in cell.index
        )
        rows.append(
            {
                "response": r,
                "effect": "beta:N",
                "ss": float(ss),
                "share_percent": 100 * float(ss) / total_ss[r],
            }
        )
    return level_means, pd.DataFrame(rows)
