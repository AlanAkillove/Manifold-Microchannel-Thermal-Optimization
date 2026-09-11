"""结构化验证方法与基准模型。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from .rsm import CubicRSM, encode, total_degree_powers


def interlaced_folds(frame: pd.DataFrame) -> np.ndarray:
    """根据各参数水平的序号划分均衡交错折。"""
    ib = frame["beta"].map({0.10: 0, 0.15: 1, 0.20: 2, 0.30: 3}).to_numpy()
    ie = frame["eta"].map({3.0: 0, 3.5: 1, 4.0: 2, 4.5: 3}).to_numpy()
    inn = frame["N"].map({2: 0, 4: 1, 6: 2, 8: 3, 10: 4}).to_numpy()
    return (ib + 2 * ie + 3 * inn) % 5


def _rmse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2, axis=0))


def polynomial_cv_predictions(frame: pd.DataFrame, degree: int, folds: np.ndarray) -> np.ndarray:
    """返回完全多项式模型的折外预测值。"""
    predictions = np.empty((len(frame), 3))
    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        model = CubicRSM.fit(
            frame.loc[train, "beta"],
            frame.loc[train, "eta"],
            frame.loc[train, "N"],
            frame.loc[train, ["R", "P", "U"]],
            degree,
        )
        predictions[test] = model.predict(
            frame.loc[test, "beta"].to_numpy(),
            frame.loc[test, "eta"].to_numpy(),
            frame.loc[test, "N"].to_numpy(),
        )
    return predictions


def polynomial_cv(frame: pd.DataFrame, degree: int, folds: np.ndarray) -> np.ndarray:
    """根据结构化折外预测计算三项响应各自的 RMSE。"""
    predictions = polynomial_cv_predictions(frame, degree, folds)
    return _rmse(frame[["R", "P", "U"]], predictions)


def polynomial_loocv(frame: pd.DataFrame, degree: int = 3) -> np.ndarray:
    powers = total_degree_powers(degree)
    x = CubicRSM.design_matrix(encode(frame.beta, frame.eta, frame.N), powers)
    y = frame[["R", "P", "U"]].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    leverage = np.einsum("ij,jk,ik->i", x, np.linalg.pinv(x.T @ x), x)
    loo_residual = residual / (1.0 - leverage[:, None])
    return np.sqrt(np.mean(loo_residual**2, axis=0))


def internal_n_holdout(frame: pd.DataFrame, degree: int = 3) -> np.ndarray:
    errors = []
    for n_rows in (4, 6, 8):
        test = frame.N.eq(n_rows).to_numpy()
        model = CubicRSM.fit(
            frame.loc[~test, "beta"],
            frame.loc[~test, "eta"],
            frame.loc[~test, "N"],
            frame.loc[~test, ["R", "P", "U"]],
            degree,
        )
        errors.append(
            frame.loc[test, ["R", "P", "U"]].to_numpy()
            - model.predict(
                frame.loc[test, "beta"].to_numpy(),
                frame.loc[test, "eta"].to_numpy(),
                frame.loc[test, "N"].to_numpy(),
            )
        )
    return _rmse(np.zeros((48, 3)), np.vstack(errors))


def local_beta_eta_block_holdout(frame: pd.DataFrame, degree: int = 3) -> np.ndarray:
    """留出中央 ``beta-eta`` 参数块并计算预测 RMSE。

    验证集由 ``beta`` 取 0.15、0.20 且 ``eta`` 取 3.5、4.0 的 20 个
    组合构成，包含全部五个已观测 ``N`` 水平。训练集中仍保留每个参数的
    全部水平，但测试区域内相邻的 ``beta-eta`` 组合整体缺失。
    """
    test = frame["beta"].isin((0.15, 0.20)) & frame["eta"].isin((3.5, 4.0))
    if int(test.sum()) != 20:
        raise ValueError("中央相邻参数留出验证要求完整的 4×4×5 有针肋样本")
    model = CubicRSM.fit(
        frame.loc[~test, "beta"],
        frame.loc[~test, "eta"],
        frame.loc[~test, "N"],
        frame.loc[~test, ["R", "P", "U"]],
        degree,
    )
    observed = frame.loc[test, ["R", "P", "U"]].to_numpy()
    predicted = model.predict(
        frame.loc[test, "beta"].to_numpy(),
        frame.loc[test, "eta"].to_numpy(),
        frame.loc[test, "N"].to_numpy(),
    )
    return _rmse(observed, predicted)


def gp_cv(frame: pd.DataFrame, kernel_name: str, folds: np.ndarray) -> np.ndarray:
    x = encode(frame.beta, frame.eta, frame.N)
    y = frame[["R", "P", "U"]].to_numpy()
    predictions = np.empty_like(y)
    base = (
        Matern(length_scale=np.ones(3), nu=2.5) if kernel_name == "Matern-5/2" else RBF(np.ones(3))
    )
    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        for j in range(3):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-8, (1e-12, 1e-2))
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=2026
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(x[train], y[train, j])
            predictions[test, j] = gp.predict(x[test])
    return _rmse(y, predictions)
