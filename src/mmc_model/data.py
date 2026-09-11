"""原始附件数据的读取与校验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["sample", "beta", "eta", "N", "R", "P", "U"]


def find_attachment2(project_root: str | Path) -> Path:
    """返回 data/raw 中唯一的附件 2 工作簿路径。"""
    raw_dir = Path(project_root) / "data" / "raw"
    matches = sorted(raw_dir.glob("attachment_2*.xlsx"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"原始数据目录 {raw_dir} 中应有且仅有一个附件 2 工作簿，实际找到 {len(matches)} 个。"
        )
    return matches[0]


def load_attachment2(path: str | Path) -> pd.DataFrame:
    """读取并校验附件 2。

    返回 84 组数值结果，其中前 4 组为无针肋基准，后 80 组构成完整的
    ``4 x 4 x 5`` 有针肋全因子设计。
    """
    frame = pd.read_excel(Path(path), header=None, skiprows=2, usecols="A:G")
    frame.columns = COLUMNS
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame["sample"] = frame["sample"].astype(int)
    frame["N"] = frame["N"].astype(int)

    if len(frame) != 84 or frame["sample"].tolist() != list(range(1, 85)):
        raise ValueError("附件 2 必须完整包含且仅包含编号 1 至 84 的样本。")
    pin = frame.query("beta > 0").copy()
    expected = {
        (b, e, n)
        for b in (0.10, 0.15, 0.20, 0.30)
        for e in (3.0, 3.5, 4.0, 4.5)
        for n in (2, 4, 6, 8, 10)
    }
    observed = set(map(tuple, pin[["beta", "eta", "N"]].to_numpy()))
    if observed != expected:
        raise ValueError("有针肋样本未覆盖预期的完整全因子设计。")
    if not np.isfinite(frame[["R", "P", "U"]].to_numpy()).all():
        raise ValueError("附件 2 的响应指标中存在非有限数值。")
    return frame


def split_topologies(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别返回无针肋基准样本和有针肋样本。"""
    return frame.query("beta == 0").copy(), frame.query("beta > 0").copy()
