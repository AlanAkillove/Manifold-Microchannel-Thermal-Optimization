"""混合整数多目标优化、权重范围计算与参数误差分析。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

from .rsm import CubicRSM


@dataclass(frozen=True)
class Design:
    """由连续几何参数和整数针肋排数组成的设计。"""

    beta: float
    eta: float
    n_rows: int


def objective_extrema(
    model: CubicRSM, n_values=range(2, 11)
) -> tuple[np.ndarray, np.ndarray, list[Design]]:
    minima = np.empty(3)
    argmins: list[Design] = []
    for j in range(3):
        best = None
        for n_rows in n_values:
            result = differential_evolution(
                lambda x: float(model.predict(x[0], x[1], n_rows)[j]),
                bounds=[(0.10, 0.30), (3.0, 4.5)],
                seed=2026 + j * 17 + n_rows,
                init="sobol",
                popsize=30,
                maxiter=1000,
                polish=True,
                tol=1e-12,
                updating="immediate",
            )
            candidate = (float(result.fun), Design(float(result.x[0]), float(result.x[1]), n_rows))
            if best is None or candidate[0] < best[0]:
                best = candidate
        minima[j], design = best
        argmins.append(design)
    # 可行域最大值只作为较保守的归一化上界；后续会根据采样得到的
    # 非支配解集合重新计算 Pareto 参考上界。
    maxima = np.full(3, -np.inf)
    for j in range(3):
        for n_rows in n_values:
            result = differential_evolution(
                lambda x: -float(model.predict(x[0], x[1], n_rows)[j]),
                bounds=[(0.10, 0.30), (3.0, 4.5)],
                seed=3026 + j * 17 + n_rows,
                init="sobol",
                popsize=30,
                maxiter=1000,
                polish=True,
                tol=1e-12,
            )
            maxima[j] = max(maxima[j], -float(result.fun))
    return minima, maxima, argmins


def evaluate_grid(
    model: CubicRSM, beta_points: int = 241, eta_points: int = 181, n_values=range(2, 11)
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linspace(0.10, 0.30, beta_points)
    eta = np.linspace(3.0, 4.5, eta_points)
    bb, ee = np.meshgrid(beta, eta, indexing="xy")
    designs, values = [], []
    for n_rows in n_values:
        nn = np.full(bb.size, n_rows)
        designs.append(np.column_stack((bb.ravel(), ee.ravel(), nn)))
        values.append(model.predict(bb.ravel(), ee.ravel(), nn))
    return np.vstack(designs), np.vstack(values)


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    """使用 Fenwick 树前缀最小值筛选三目标最小化问题的非支配解。"""
    values = np.asarray(values)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    p_rank = np.unique(values[:, 1], return_inverse=True)[1]
    tree = np.full(p_rank.max() + 2, np.inf)
    keep = np.zeros(len(values), dtype=bool)

    def query(idx: int) -> float:
        result = np.inf
        idx += 1
        while idx > 0:
            result = min(result, tree[idx])
            idx -= idx & -idx
        return result

    def update(idx: int, val: float) -> None:
        idx += 1
        while idx < len(tree):
            tree[idx] = min(tree[idx], val)
            idx += idx & -idx

    for idx in order:
        prior_u = query(int(p_rank[idx]))
        # 若某点在前两个目标上已被支配，第三个目标相等也不能使其成为非支配解。
        if prior_u > values[idx, 2] + 1e-12:
            keep[idx] = True
        update(int(p_rank[idx]), float(values[idx, 2]))
    return keep


def normalized(values: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    return (np.asarray(values) - ideal) / (nadir - ideal)


def chebyshev_compromise(
    model: CubicRSM, ideal: np.ndarray, nadir: np.ndarray, n_values=range(2, 11)
) -> tuple[Design, np.ndarray, np.ndarray]:
    best = None
    scale = nadir - ideal
    for n_rows in n_values:

        def merit(x):
            z = (model.predict(x[0], x[1], n_rows) - ideal) / scale
            return float(np.max(z) + 1e-7 * np.sum(z))

        result = differential_evolution(
            merit,
            [(0.10, 0.30), (3.0, 4.5)],
            seed=4026 + n_rows,
            init="sobol",
            popsize=24,
            maxiter=800,
            tol=1e-12,
            polish=True,
        )
        f = model.predict(result.x[0], result.x[1], n_rows)
        z = (f - ideal) / scale
        candidate = (
            float(np.max(z)),
            float(np.sum(z)),
            Design(float(result.x[0]), float(result.x[1]), n_rows),
            f,
            z,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], best[3], best[4]


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * abs(
        np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
    )


def clip_halfplane(
    poly: np.ndarray, a: float, b: float, c: float, tol: float = 1e-12
) -> np.ndarray:
    """用半平面 ``a*u + b*v <= c`` 裁剪凸多边形。"""
    if len(poly) == 0:
        return poly
    out = []
    for start, end in zip(poly, np.roll(poly, -1, axis=0)):
        fs = a * start[0] + b * start[1] - c
        fe = a * end[0] + b * end[1] - c
        inside_s, inside_e = fs <= tol, fe <= tol
        if inside_s:
            out.append(start)
        if inside_s != inside_e:
            t = fs / (fs - fe)
            out.append(start + t * (end - start))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def weight_acceptability(z_values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """计算二维权重三角形内各方案最优区域的精确面积。

    在线性加权下，只有凸包顶点可能对应面积为正的最优权重区域。每个返回值
    等于相应多边形面积除以整个权重三角形面积，因此除数值舍入误差外，各值
    之和为 1。
    """
    z_values = np.asarray(z_values)
    hull_indices = np.unique(ConvexHull(z_values, qhull_options="QJ").simplices)
    areas = np.zeros(len(z_values))
    polygons = [np.empty((0, 2)) for _ in range(len(z_values))]
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triangle_area = polygon_area(triangle)
    hull_values = z_values[hull_indices]
    for local_i, global_i in enumerate(hull_indices):
        poly = triangle.copy()
        zi = hull_values[local_i]
        for zj in hull_values:
            d = zi - zj
            poly = clip_halfplane(poly, d[0] - d[2], d[1] - d[2], -d[2])
            if len(poly) == 0:
                break
        area = polygon_area(poly) / triangle_area
        if area > 1e-10:
            areas[global_i] = area
            polygons[global_i] = poly
    return areas, polygons


def local_dimensionless_sensitivity(model: CubicRSM, design: Design) -> np.ndarray:
    f = model.predict(design.beta, design.eta, design.n_rows)
    gradient = model.gradient(design.beta, design.eta, design.n_rows)[:, :2]
    x = np.array([design.beta, design.eta])
    return np.abs(gradient * x[None, :] / f[:, None])


def worst_case_box(
    model: CubicRSM, design: Design, relative_radius: float, baseline: np.ndarray
) -> dict[str, np.ndarray]:
    """在可行域截取后的参数误差区间内分别求各响应的最大值。"""
    lo = np.maximum([0.10, 3.0], np.array([design.beta, design.eta]) * (1 - relative_radius))
    hi = np.minimum([0.30, 4.5], np.array([design.beta, design.eta]) * (1 + relative_radius))
    worst = np.empty(3)
    locations = np.empty((3, 2))
    for j in range(3):
        result = differential_evolution(
            lambda x: -float(model.predict(x[0], x[1], design.n_rows)[j]),
            list(zip(lo, hi)),
            seed=5026 + j,
            tol=1e-11,
            polish=True,
        )
        worst[j] = -result.fun
        locations[j] = result.x
    degradation = (worst - baseline) / baseline
    return {
        "lower": lo,
        "upper": hi,
        "worst": worst,
        "relative_degradation": degradation,
        "locations": locations,
    }
