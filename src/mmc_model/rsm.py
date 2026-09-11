"""完全多项式响应面模型。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


def total_degree_powers(degree: int, dimensions: int = 3) -> np.ndarray:
    """生成先按总次数、再按字典序排列的指数元组。"""
    powers = [p for p in product(range(degree + 1), repeat=dimensions) if sum(p) <= degree]
    powers.sort(key=lambda p: (sum(p), p))
    return np.asarray(powers, dtype=int)


def encode(beta: np.ndarray, eta: np.ndarray, n_rows: np.ndarray) -> np.ndarray:
    """将物理变量映射为论文使用的中心化编码变量。"""
    return np.column_stack(
        (
            (np.asarray(beta) - 0.20) / 0.10,
            (np.asarray(eta) - 3.75) / 0.75,
            (np.asarray(n_rows) - 6.0) / 4.0,
        )
    )


@dataclass
class CubicRSM:
    """用于 ``R*``、``P*`` 和 ``U*`` 的完全三次响应面。"""

    powers: np.ndarray
    coefficients: np.ndarray
    responses: tuple[str, ...] = ("R", "P", "U")

    @classmethod
    def fit(
        cls, beta: np.ndarray, eta: np.ndarray, n_rows: np.ndarray, y: np.ndarray, degree: int = 3
    ) -> CubicRSM:
        powers = total_degree_powers(degree)
        design = cls.design_matrix(encode(beta, eta, n_rows), powers)
        coef, *_ = np.linalg.lstsq(design, np.asarray(y), rcond=None)
        return cls(powers=powers, coefficients=coef)

    @staticmethod
    def design_matrix(x: np.ndarray, powers: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.prod(x[:, None, :] ** powers[None, :, :], axis=2)

    def predict(
        self, beta: np.ndarray | float, eta: np.ndarray | float, n_rows: np.ndarray | float
    ) -> np.ndarray:
        beta, eta, n_rows = np.broadcast_arrays(beta, eta, n_rows)
        shape = beta.shape
        x = encode(beta.ravel(), eta.ravel(), n_rows.ravel())
        pred = self.design_matrix(x, self.powers) @ self.coefficients
        return pred.reshape(shape + (len(self.responses),))

    def gradient(self, beta: float, eta: float, n_rows: float) -> np.ndarray:
        """以 ``3 x 3`` 数组返回 ``d(R,P,U)/d(beta,eta,N)``。"""
        x = encode(np.array([beta]), np.array([eta]), np.array([n_rows]))[0]
        scales = np.array([0.10, 0.75, 4.0])
        grad = np.zeros((len(self.responses), 3))
        for k in range(3):
            active = self.powers[:, k] > 0
            reduced = self.powers[active].copy()
            factors = reduced[:, k].astype(float)
            reduced[:, k] -= 1
            monomials = np.prod(x[None, :] ** reduced, axis=1)
            grad[:, k] = (factors * monomials) @ self.coefficients[active] / scales[k]
        return grad

    @property
    def condition_number(self) -> float:
        raise AttributeError(
            "条件数取决于拟合所用的试验设计，请调用 design_condition_number()。"
        )


def design_condition_number(
    beta: np.ndarray, eta: np.ndarray, n_rows: np.ndarray, degree: int = 3
) -> float:
    powers = total_degree_powers(degree)
    matrix = CubicRSM.design_matrix(encode(beta, eta, n_rows), powers)
    return float(np.linalg.cond(matrix))
