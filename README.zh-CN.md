# 芯片歧管式微通道热管理系统建模与优化

[English README](README.md)

本项目整理一次芯片歧管式微通道热管理系统建模任务的论文、计算代码、数据和图件，提供问题一至问题五的复算流程。本任务为建模练习，不对应任何官方年度竞赛题号。

[论文 PDF](docs/paper/tex/main.pdf) · [AI 工具使用详情](docs/paper/tex/AI工具使用详情.pdf) · [结果 JSON](outputs/results.json) · [复现说明](REPRODUCIBILITY.md) · [许可证](LICENSE)

![芯片歧管式微通道系统结构](outputs/figures/fig_system_structure.png)

## 项目亮点

- 从流量分配、针肋换热以及芯片和衬底中的热传导出发解释结构参数的作用；
- 比较二次、三次响应面和高斯过程模型，并使用多种交叉验证方法检查代理模型；
- 在针肋排数为整数的约束下进行多目标 Pareto 优化；
- 在权重单纯形中计算方案的可接受区域，分析偏好变化下的稳健设计；
- 提供参数扰动分析、机器可读结果和回归测试。

## 主要结果

| 项目 | 结果 |
| --- | --- |
| 代理模型验证 | 三次响应面的五折 RMSE 对 `R*`、`P*`、`U*` 分别为 `3.46e-7`、`5.59e-4`、`3.00e-4` |
| 名义折中方案 | Chebyshev 准则得到 `beta = 0.22494`、`eta = 4.50`、`N = 6` |
| 偏好稳健区域 | `beta in [0.215, 0.224]`、`eta = 4.50`、`N = 3-4`；加密网格覆盖率为 `49.99%` |
| 网格稳定性 | `401 x 301` 加密网格仍得到 `N = 6` 的名义方案 |
| 参数扰动 | 最大增幅由压降控制，两种代表方案在不超过 `+/-5%` 扰动下分别为 `7.10%` 和 `6.62%` |

![响应面结果](outputs/figures/fig_q2_response_surfaces.png)

## 方法概述

程序读取 84 组数据，其中包括 4 组无针肋基准样本和 `4 x 4 x 5` 组有针肋全因子样本。首先利用守恒关系和尺度分析解释系统机理；随后比较多项式响应面和高斯过程模型，并进行交错五折、留一、按 `N` 水平留出以及中央 `beta-eta` 区块留出验证；最后枚举整数针肋排数，搜索连续结构参数，划分权重单纯形并分析参数扰动。

## 复现

需要 Python 3.10 或更高版本。建议使用虚拟环境：

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

先运行快速检查：

    python scripts/run_analysis.py --quick
    python -m pytest -q

快速检查只读取原始数据、拟合模型并运行小网格，不覆盖发布结果。完整分析会更新整理后的数据、17 个结果表和 `outputs/results.json`：

    python scripts/run_analysis.py --full
    python scripts/make_figures.py
    python -m pytest -q

快速检查通常在 15 秒内完成；完整分析在普通桌面 CPU 上约需 10–15 分钟，其中问题四和问题五的优化搜索最耗时。随机优化步骤使用固定随机种子。

## 目录结构

    data/raw/                 原始题目和附件
    data/processed/           由原始附件整理得到的机器可读数据
    docs/paper/tex/           论文源文件、参考文献、模板和 PDF
    outputs/figures/          论文图件及 PowerPoint 绘图源文件
    outputs/tables/           各问题的机器可读结果表
    outputs/results.json      主要模型、优化和稳定性结果
    scripts/                  完整分析和绘图入口
    src/mmc_model/            可复用的模型、验证和优化函数
    tests/                    回归测试

## 数据、论文和第三方材料

`data/raw/` 中的文件是计算使用的原始输入，保留这些文件是为了支持结果复现，但它们不属于代码的 MIT License 许可范围。论文所需的 LaTeX 模板、参考文献样式和字体也具有各自的来源与使用条件，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

论文 PDF 为 [docs/paper/tex/main.pdf](docs/paper/tex/main.pdf)，源文件为 [docs/paper/tex/main.tex](docs/paper/tex/main.tex)。引用信息保存在 [references.bib](docs/paper/tex/references.bib) 中，参考文献全文和本地写作过程文件不纳入仓库。

## 许可证

可复用代码按 [MIT License](LICENSE) 发布。题目附件、引用文献、LaTeX 模板、参考文献样式和字体仍分别适用其原有条款。
