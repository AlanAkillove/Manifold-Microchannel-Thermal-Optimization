#!/usr/bin/env python3
"""运行全部数值计算，并写出论文所需的机器可读结果。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmc_model.data import find_attachment2, load_attachment2, split_topologies  # noqa: E402
from mmc_model.optimization import (  # noqa: E402
    Design,
    chebyshev_compromise,
    evaluate_grid,
    local_dimensionless_sensitivity,
    nondominated_mask,
    normalized,
    objective_extrema,
    weight_acceptability,
    worst_case_box,
)
from mmc_model.q1 import (  # noqa: E402
    coolant_energy_balance,
    factorial_effects,
    mechanism_pressure_fit,
    no_pin_pressure_fit,
)
from mmc_model.rsm import CubicRSM, design_condition_number  # noqa: E402
from mmc_model.validation import (  # noqa: E402
    gp_cv,
    interlaced_folds,
    internal_n_holdout,
    local_beta_eta_block_holdout,
    polynomial_cv,
    polynomial_cv_predictions,
    polynomial_loocv,
)


def serializable(value):
    """将 NumPy 数值和设计对象转换为可写入 JSON 的 Python 对象。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Design):
        return {"beta": value.beta, "eta": value.eta, "N": value.n_rows}
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def l2_compromise(
    model: CubicRSM, ideal: np.ndarray, nadir: np.ndarray, n_values=range(2, 11)
) -> tuple[float, Design, np.ndarray, np.ndarray]:
    """寻找欧氏距离意义下最接近理想点的可行设计。"""
    best = None
    for n_rows in n_values:
        result = differential_evolution(
            lambda x: float(
                np.linalg.norm(normalized(model.predict(x[0], x[1], n_rows), ideal, nadir))
            ),
            [(0.10, 0.30), (3.0, 4.5)],
            seed=6026 + n_rows,
            init="sobol",
            popsize=24,
            maxiter=800,
            tol=1e-12,
            polish=True,
        )
        f = model.predict(result.x[0], result.x[1], n_rows)
        z = normalized(f, ideal, nadir)
        candidate = (
            float(np.linalg.norm(z)),
            Design(float(result.x[0]), float(result.x[1]), n_rows),
            f,
            z,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def q4_acceptability_study(
    model: CubicRSM,
    ideal: np.ndarray,
    nadir: np.ndarray,
    beta_points: int,
    eta_points: int,
    n_values,
) -> tuple[pd.DataFrame, dict]:
    """在指定候选网格上重新计算问题四的最优权重区域。"""
    n_values = tuple(n_values)
    designs, values = evaluate_grid(
        model, beta_points=beta_points, eta_points=eta_points, n_values=n_values
    )
    mask = nondominated_mask(values)
    pareto_designs = designs[mask]
    pareto_values = values[mask]
    z_values = normalized(pareto_values, ideal, nadir)
    areas, _ = weight_acceptability(z_values)
    table = pd.DataFrame(
        np.column_stack((pareto_designs, pareto_values, areas)),
        columns=["beta", "eta", "N", "R", "P", "U", "acceptability"],
    )
    table["N"] = table["N"].astype(int)
    robust_idx = int(np.argmax(areas))
    by_n = {
        str(n_rows): float(table.loc[table["N"].eq(n_rows), "acceptability"].sum())
        for n_rows in n_values
    }
    summary = {
        "beta_points": beta_points,
        "eta_points": eta_points,
        "beta_step": 0.20 / (beta_points - 1),
        "eta_step": 1.50 / (eta_points - 1),
        "N": list(n_values),
        "pareto_grid_count": int(len(table)),
        "positive_weight_regions": int(np.count_nonzero(areas)),
        "robust_design": Design(
            float(pareto_designs[robust_idx, 0]),
            float(pareto_designs[robust_idx, 1]),
            int(pareto_designs[robust_idx, 2]),
        ),
        "objectives": pareto_values[robust_idx],
        "acceptability": float(areas[robust_idx]),
        "acceptability_by_N": by_n,
    }
    return table, summary


def q4_cluster_profile(
    table: pd.DataFrame, eta: float, n_values: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """按 beta 汇总给定 eta 和排数组合的权重面积，并返回前缀和。"""
    selected = table[np.isclose(table["eta"], eta) & table["N"].isin(n_values)]
    grouped = selected.groupby("beta")["acceptability"].sum().sort_index()
    beta_values = grouped.index.to_numpy(dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(grouped.to_numpy(dtype=float))))
    return beta_values, cumulative


def q4_interval_acceptability(
    profile: tuple[np.ndarray, np.ndarray], beta_interval: tuple[float, float]
) -> float:
    """由前缀和计算连续 beta 区间对应的权重面积。"""
    beta_values, cumulative = profile
    lower, upper = beta_interval
    left = int(np.searchsorted(beta_values, lower, side="left"))
    right = int(np.searchsorted(beta_values, upper, side="right"))
    return float(cumulative[right] - cumulative[left])


def q4_cluster_definition(
    main_table: pd.DataFrame,
    check_tables: tuple[pd.DataFrame, ...],
    coverage_target: float = 0.50,
) -> tuple[dict, list[float]]:
    """由主网格确定问题四的邻近设计区域，并在其他网格上复算其面积。

    先在主网格中选择总权重面积最大的相邻排数组合，再选择该组合面积最大的
    ``eta`` 水平。随后仅依据主网格寻找最窄的连续 ``beta`` 区间，使其包含
    主网格的单点最优设计，且累计面积达到给定比例。区域确定后保持边界不变，
    再计算其他离散设置下的覆盖面积。
    """
    by_n = main_table.groupby("N")["acceptability"].sum()
    observed_n = sorted(int(value) for value in by_n.index)
    adjacent_pairs = [(n_rows, n_rows + 1) for n_rows in observed_n if n_rows + 1 in by_n.index]
    n_values = max(adjacent_pairs, key=lambda pair: float(by_n.loc[list(pair)].sum()))

    pair_rows = main_table[main_table["N"].isin(n_values)]
    eta = float(pair_rows.groupby("eta")["acceptability"].sum().idxmax())

    peak_beta = float(main_table.loc[main_table["acceptability"].idxmax(), "beta"])
    main_profile = q4_cluster_profile(main_table, eta, n_values)
    beta_nodes = np.sort(main_table["beta"].unique())
    best: tuple[float, float, float] | None = None
    for lower_index, lower in enumerate(beta_nodes):
        for upper in beta_nodes[lower_index:]:
            if not lower <= peak_beta <= upper:
                continue
            interval = (float(lower), float(upper))
            coverage = q4_interval_acceptability(main_profile, interval)
            if coverage + 1e-12 < coverage_target:
                continue
            candidate = (float(upper - lower), float(lower), float(upper))
            if best is None or candidate < best:
                best = candidate
            break
    if best is None:
        raise RuntimeError("没有找到满足权重覆盖要求的连续 beta 区间")

    _, lower, upper = best
    lower, upper = round(lower, 12), round(upper, 12)
    interval = (lower, upper)
    profiles = [
        q4_cluster_profile(table, eta, n_values)
        for table in (main_table, *check_tables)
    ]
    coverages = [q4_interval_acceptability(profile, interval) for profile in profiles]
    definition = {
        "beta": [lower, upper],
        "eta": eta,
        "N": list(n_values),
        "coverage_target": coverage_target,
    }
    return definition, coverages


def main() -> None:
    workbook = find_attachment2(ROOT)
    output = ROOT / "outputs"
    tables = output / "tables"
    data_dir = ROOT / "data" / "processed"
    tables.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = load_attachment2(workbook)
    no_pin, pin = split_topologies(frame)
    frame.to_csv(data_dir / "attachment2_tidy.csv", index=False, float_format="%.9g")

    q1 = {
        "energy_balance": coolant_energy_balance(),
        "no_pin_pressure_fit": no_pin_pressure_fit(no_pin),
        # 对 4 组无针肋基准，阻塞项 N*beta^2 为零，因此可将其纳入问题一的
        # 简化压降关系。问题二要求样本拓扑一致，故响应面仍只使用有针肋样本。
        "mechanism_pressure_fit": mechanism_pressure_fit(frame),
    }
    level_means, effects = factorial_effects(pin)
    level_means.to_csv(tables / "q1_level_means.csv", index=False, float_format="%.9g")
    effects.to_csv(tables / "q1_effect_shares.csv", index=False, float_format="%.9g")

    model = CubicRSM.fit(pin.beta, pin.eta, pin.N, pin[["R", "P", "U"]])
    fit_pred = model.predict(pin.beta.to_numpy(), pin.eta.to_numpy(), pin.N.to_numpy())
    fitted = pin.copy()
    for j, response in enumerate(("R", "P", "U")):
        fitted[f"{response}_pred"] = fit_pred[:, j]
        fitted[f"{response}_resid"] = fitted[response] - fit_pred[:, j]
    coef = pd.DataFrame(model.powers, columns=["power_beta", "power_eta", "power_N"])
    coef[["R", "P", "U"]] = model.coefficients
    coef.to_csv(tables / "q2_cubic_coefficients.csv", index=False, float_format="%.12g")

    folds = interlaced_folds(pin)
    cv_pred = polynomial_cv_predictions(pin, degree=3, folds=folds)
    for j, response in enumerate(("R", "P", "U")):
        fitted[f"{response}_cv_pred"] = cv_pred[:, j]
        fitted[f"{response}_cv_resid"] = fitted[response] - cv_pred[:, j]
    fitted.to_csv(tables / "q2_fitted_values.csv", index=False, float_format="%.12g")

    observed = pin[["R", "P", "U"]].to_numpy()
    residual = observed - fit_pred
    training_rmse = np.sqrt(np.mean(residual**2, axis=0))
    training_mae = np.mean(np.abs(residual), axis=0)
    training_r2 = 1.0 - np.sum(residual**2, axis=0) / np.sum(
        (observed - observed.mean(axis=0)) ** 2, axis=0
    )
    fit_metrics = pd.DataFrame(
        {
            "response": ("R", "P", "U"),
            "R2": training_r2,
            "RMSE": training_rmse,
            "MAE": training_mae,
        }
    )
    fit_metrics.to_csv(tables / "q2_fit_metrics.csv", index=False, float_format="%.12g")

    validation_rows = []
    # 参数 beta 和 eta 均只有 4 个水平；若不引入正则化，完整四次项无法独立识别。
    for degree in (2, 3):
        validation_rows.append(
            {
                "model": f"complete polynomial d={degree}",
                **dict(zip(("R", "P", "U"), polynomial_cv(pin, degree, folds))),
            }
        )
    for kernel in ("Matern-5/2", "RBF"):
        validation_rows.append(
            {
                "model": f"Gaussian process {kernel}",
                **dict(zip(("R", "P", "U"), gp_cv(pin, kernel, folds))),
            }
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(tables / "q2_model_validation.csv", index=False, float_format="%.12g")
    loocv_rmse = polynomial_loocv(pin)
    n_holdout_rmse = internal_n_holdout(pin)
    block_holdout_rmse = local_beta_eta_block_holdout(pin)
    fivefold_rmse = np.array([validation_rows[1][key] for key in ("R", "P", "U")])
    pd.DataFrame(
        [
            {"validation": "interlaced 5-fold", **dict(zip(("R", "P", "U"), fivefold_rmse))},
            {"validation": "leave-one-out", **dict(zip(("R", "P", "U"), loocv_rmse))},
            {
                "validation": "internal N-level holdout",
                **dict(zip(("R", "P", "U"), n_holdout_rmse)),
            },
            {
                "validation": "central beta-eta block holdout",
                **dict(zip(("R", "P", "U"), block_holdout_rmse)),
            },
        ]
    ).to_csv(tables / "q2_prediction_checks.csv", index=False, float_format="%.12g")

    # 检查压降随歧管深高比增大而减小这一物理预期。
    beta_check = np.linspace(0.10, 0.30, 81)
    eta_check = np.linspace(3.0, 4.5, 81)
    derivative_p_eta = []
    for n_rows in range(2, 11):
        for beta in beta_check:
            for eta in eta_check:
                derivative_p_eta.append(model.gradient(beta, eta, n_rows)[1, 1])

    ideal, feasible_max, ideal_designs = objective_extrema(model)
    designs, values = evaluate_grid(model, beta_points=201, eta_points=151)
    pareto_mask = nondominated_mask(values)
    pareto_designs = designs[pareto_mask]
    pareto_values = values[pareto_mask]
    pareto_nadir = pareto_values.max(axis=0)
    pareto = pd.DataFrame(
        np.column_stack((pareto_designs, pareto_values)),
        columns=["beta", "eta", "N", "R", "P", "U"],
    )
    pareto["N"] = pareto["N"].astype(int)
    pareto.to_csv(tables / "q3_pareto_grid.csv", index=False, float_format="%.10g")

    q3_design, q3_f, q3_z = chebyshev_compromise(model, ideal, pareto_nadir)
    even_design, even_f, even_z = chebyshev_compromise(model, ideal, pareto_nadir, (2, 4, 6, 8, 10))
    l2 = l2_compromise(model, ideal, pareto_nadir)

    pareto_z = normalized(pareto_values, ideal, pareto_nadir)
    areas, polygons = weight_acceptability(pareto_z)
    pareto["acceptability"] = areas
    pareto.to_csv(tables / "q4_weight_acceptability.csv", index=False, float_format="%.12g")
    acceptability_by_n = {
        str(n_rows): float(pareto.loc[pareto["N"].eq(n_rows), "acceptability"].sum())
        for n_rows in range(2, 11)
    }
    robust_idx = int(np.argmax(areas))
    robust_design = Design(
        float(pareto_designs[robust_idx, 0]),
        float(pareto_designs[robust_idx, 1]),
        int(pareto_designs[robust_idx, 2]),
    )
    robust_f = pareto_values[robust_idx]
    robust_z = pareto_z[robust_idx]
    polygon_rows = []
    for idx, polygon in enumerate(polygons):
        if len(polygon):
            for vertex, (w_r, w_p) in enumerate(polygon):
                polygon_rows.append(
                    {
                        "candidate": idx,
                        "vertex": vertex,
                        "w_R": w_r,
                        "w_P": w_p,
                        "w_U": 1 - w_r - w_p,
                        "acceptability": areas[idx],
                    }
                )
    pd.DataFrame(polygon_rows).to_csv(
        tables / "q4_weight_polygons.csv", index=False, float_format="%.12g"
    )

    # 检查奇数 N 插值和连续参数网格间距对问题四结果的影响。
    q4_even_table, q4_even = q4_acceptability_study(
        model, ideal, pareto_nadir, 201, 151, (2, 4, 6, 8, 10)
    )
    q4_even_table.to_csv(tables / "q4_even_n_check.csv", index=False, float_format="%.12g")
    q4_fine_table, q4_fine = q4_acceptability_study(
        model, ideal, pareto_nadir, 401, 301, range(2, 11)
    )
    q4_fine_table.to_csv(tables / "q4_fine_grid_check.csv", index=False, float_format="%.12g")

    cluster_definition, cluster_coverages = q4_cluster_definition(
        pareto, (q4_even_table, q4_fine_table)
    )
    q4_cluster = {
        "definition": cluster_definition,
        "main_grid": cluster_coverages[0],
        "even_N_only": cluster_coverages[1],
        "fine_grid": cluster_coverages[2],
    }
    pd.DataFrame(
        [
            {"calculation": key, "acceptability": value}
            for key, value in q4_cluster.items()
            if key != "definition"
        ]
    ).to_csv(tables / "q4_cluster_stability.csv", index=False, float_format="%.12g")

    # 复用加密网格 Pareto 集，检查问题三的归一化参考值和 Chebyshev 方案
    # 是否依赖原始网格间距。
    q3_fine_nadir = q4_fine_table[["R", "P", "U"]].max().to_numpy()
    q3_fine_design, q3_fine_f, q3_fine_z = chebyshev_compromise(model, ideal, q3_fine_nadir)
    q3_fine = {
        "beta_points": 401,
        "eta_points": 301,
        "beta_step": 0.0005,
        "eta_step": 0.005,
        "N": list(range(2, 11)),
        "pareto_grid_count": int(len(q4_fine_table)),
        "pareto_reference_upper": q3_fine_nadir,
        "chebyshev": {
            "design": q3_fine_design,
            "objectives": q3_fine_f,
            "normalized": q3_fine_z,
            "D_inf": float(np.max(q3_fine_z)),
        },
    }
    pd.DataFrame(
        [
            {
                "grid": "main",
                "beta_step": 0.001,
                "eta_step": 0.01,
                "pareto_count": len(pareto),
                "fref_R": pareto_nadir[0],
                "fref_P": pareto_nadir[1],
                "fref_U": pareto_nadir[2],
                "beta": q3_design.beta,
                "eta": q3_design.eta,
                "N": q3_design.n_rows,
                "R": q3_f[0],
                "P": q3_f[1],
                "U": q3_f[2],
                "D_inf": np.max(q3_z),
            },
            {
                "grid": "fine",
                "beta_step": 0.0005,
                "eta_step": 0.005,
                "pareto_count": len(q4_fine_table),
                "fref_R": q3_fine_nadir[0],
                "fref_P": q3_fine_nadir[1],
                "fref_U": q3_fine_nadir[2],
                "beta": q3_fine_design.beta,
                "eta": q3_fine_design.eta,
                "N": q3_fine_design.n_rows,
                "R": q3_fine_f[0],
                "P": q3_fine_f[1],
                "U": q3_fine_f[2],
                "D_inf": np.max(q3_fine_z),
            },
        ]
    ).to_csv(tables / "q3_fine_grid_check.csv", index=False, float_format="%.12g")

    q5_rows = []
    q5_overall_rows = []
    q5_detail = {}
    for label, design, baseline in (
        ("Q3 nominal", q3_design, q3_f),
        ("Q4 preference-robust", robust_design, robust_f),
    ):
        sensitivity = local_dimensionless_sensitivity(model, design)
        q5_detail[label] = {
            "design": design,
            "baseline": baseline,
            "local_sensitivity_rows_R_P_U_cols_beta_eta": sensitivity,
        }
        for radius in (0.01, 0.03, 0.05):
            result = worst_case_box(model, design, radius, baseline)
            q5_detail[label][f"box_{int(radius * 100)}pct"] = result
            controlling_response = int(np.argmax(result["relative_degradation"]))
            q5_overall_rows.append(
                {
                    "design": label,
                    "radius_percent": radius * 100,
                    "maximum_degradation_percent": 100
                    * result["relative_degradation"][controlling_response],
                    "controlling_response": ("R", "P", "U")[controlling_response],
                }
            )
            for j, response in enumerate(("R", "P", "U")):
                q5_rows.append(
                    {
                        "design": label,
                        "radius_percent": radius * 100,
                        "response": response,
                        "baseline": baseline[j],
                        "worst": result["worst"][j],
                        "degradation_percent": 100 * result["relative_degradation"][j],
                        "worst_beta": result["locations"][j, 0],
                        "worst_eta": result["locations"][j, 1],
                    }
                )
    pd.DataFrame(q5_rows).to_csv(tables / "q5_worst_case.csv", index=False, float_format="%.12g")
    pd.DataFrame(q5_overall_rows).to_csv(
        tables / "q5_overall_maximum.csv", index=False, float_format="%.12g"
    )

    baseline_rise = q1["energy_balance"]["mean_coolant_rise_K"]
    flow_scenarios = pd.DataFrame(
        {
            "mass_flow_ratio": (0.9, 1.0, 1.1),
            "mean_coolant_rise_K": (baseline_rise / 0.9, baseline_rise, baseline_rise / 1.1),
        }
    )
    flow_scenarios["change_percent"] = 100 * (
        flow_scenarios["mean_coolant_rise_K"] / baseline_rise - 1
    )
    flow_scenarios.to_csv(
        tables / "q5_flow_scenarios.csv", index=False, float_format="%.12g"
    )

    summary = {
        "q1": q1,
        "q2": {
            "condition_number": design_condition_number(pin.beta, pin.eta, pin.N),
            "training_R2": training_r2,
            "training_RMSE": training_rmse,
            "training_MAE": training_mae,
            "interlaced_5fold": validation_rows,
            "loocv_degree3": loocv_rmse,
            "internal_N_holdout_degree3": n_holdout_rmse,
            "local_beta_eta_block_holdout_degree3": block_holdout_rmse,
            "pressure_dP_deta_range": [min(derivative_p_eta), max(derivative_p_eta)],
        },
        "q3": {
            "ideal": ideal,
            "feasible_max": feasible_max,
            "ideal_designs": ideal_designs,
            "pareto_reference_upper": pareto_nadir,
            "pareto_grid_count": len(pareto),
            "pareto_grid": {
                "beta_points": 201,
                "eta_points": 151,
                "beta_step": 0.001,
                "eta_step": 0.01,
                "N": "integers 2--10",
            },
            "chebyshev": {
                "design": q3_design,
                "objectives": q3_f,
                "normalized": q3_z,
                "D_inf": float(np.max(q3_z)),
            },
            "even_N_only": {
                "design": even_design,
                "objectives": even_f,
                "normalized": even_z,
                "D_inf": float(np.max(even_z)),
            },
            "euclidean": {
                "score": l2[0],
                "design": l2[1],
                "objectives": l2[2],
                "normalized": l2[3],
            },
            "fine_grid_check": q3_fine,
        },
        "q4": {
            "manufacturing_grid": {"beta_step": 0.001, "eta_step": 0.01, "N": "integers 2--10"},
            "positive_weight_regions": int(np.count_nonzero(areas)),
            "robust_design": robust_design,
            "objectives": robust_f,
            "normalized": robust_z,
            "acceptability": areas[robust_idx],
            "acceptability_by_N": acceptability_by_n,
            "even_N_check": q4_even,
            "fine_grid_check": q4_fine,
            "stable_cluster": q4_cluster,
        },
        "q5": {
            **q5_detail,
            "overall_maximum": q5_overall_rows,
            "flow_scenarios": flow_scenarios.to_dict(orient="records"),
            "solver": {
                "method": "differential_evolution",
                "seed_by_response": [5026, 5027, 5028],
                "tolerance": 1e-11,
                "polish": True,
            },
        },
    }
    (output / "results.json").write_text(
        json.dumps(serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            serializable({"q2": summary["q2"], "q3": summary["q3"], "q4": summary["q4"]}),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
