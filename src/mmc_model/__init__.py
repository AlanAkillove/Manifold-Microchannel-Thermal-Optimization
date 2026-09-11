"""歧管式微通道热管理问题的模型包。"""

from .data import find_attachment2, load_attachment2
from .rsm import CubicRSM

__all__ = ["CubicRSM", "find_attachment2", "load_attachment2"]
