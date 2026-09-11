# 芯片歧管式微通道热管理系统建模与优化

本项目整理一次芯片歧管式微通道热管理系统建模任务的论文、计算代码、数据和图件，提供问题一至问题五的复算流程。本任务为建模练习，不对应任何官方年度竞赛题号。

## 研究内容

- 用流量守恒、局部换热和二维导热关系解释结构参数的作用；
- 用三次响应面预测无量纲热阻、压降和温度非均匀性；
- 枚举整数针肋排数并搜索连续参数，筛选 Pareto 方案；
- 在权重三角形中计算各候选方案的最优权重范围，并按统一规则确定邻近设计区域；
- 比较代表方案对几何误差的单项和总体最大增幅，并给出质量流量变化下的冷却液温升情景。

全部定量结论均限制在附件 2 的数据范围内：

- `0.10 <= beta <= 0.30`；
- `3.0 <= eta <= 4.5`；
- 代理模型的已观测针肋排数为 `N = 2, 4, 6, 8, 10`，优化阶段另对 `N = 3, 5, 7, 9` 作区间内插值并进行偶数排数复算检查；
- 4 个无针肋样本仅用于问题一的基准分析，不并入有针肋响应面。

## 目录结构

```text
项目根目录/
├─ data/
│  ├─ raw/                # 建模任务提供的原始附件
│  └─ processed/          # 由原始附件整理得到的机器可读数据
├─ docs/
│  └─ paper/tex/          # main.tex、参考文献库、模板与最终 PDF
├─ outputs/
│  ├─ figures/            # 论文图件及 PowerPoint 绘图源文件
│  ├─ tables/             # 各问题的机器可读结果表
│  └─ results.json        # 主要模型、优化与敏感性结果
├─ scripts/
│  ├─ run_analysis.py     # 数据核对、建模、验证和优化
│  └─ make_figures.py     # 生成数据图
├─ src/mmc_model/         # 可复用的模型与计算函数
└─ tests/                 # 回归测试
```

本地检查、渲染和写作过程文件不属于发布内容，已移出项目目录保存。

## 环境安装

需要 Python 3.10 或更高版本。建议在虚拟环境中安装：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

主要依赖包括 NumPy、Pandas、SciPy、scikit-learn、Matplotlib 和 openpyxl。

## 复现实验

在项目根目录依次执行：

```powershell
python scripts/run_analysis.py
python scripts/make_figures.py
python -m pytest -q
```

第一条命令读取 `data/raw/attachment_2_results.xlsx`，更新 `data/processed/`、`outputs/tables/` 和 `outputs/results.json`。第二条命令根据这些结果重新生成问题一至问题五的数据图。所有随机优化步骤均设置固定种子。

## 编译论文

论文源文件为 `docs/paper/tex/main.tex`。在该目录执行：

```powershell
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

也可在已安装 Perl 和 `latexmk` 的环境中执行：

```powershell
latexmk main.tex
```

最终论文输出为 `docs/paper/tex/main.pdf`。分析流程图、系统结构图和问题一模型示意图由 PowerPoint 绘制，其余数据图由 `scripts/make_figures.py` 生成。

## 数据与结果说明

- `data/raw/` 中的文件是唯一原始数据来源，程序不会修改这些文件；
- `data/processed/attachment2_tidy.csv` 是附件 2 的整理副本；
- `data/raw/attachment_1_structure.pdf` 和 `data/raw/problem_statement.pdf` 用于查阅题目和结构说明；
- `outputs/tables/` 保存论文表格、中央相邻参数留出验证、邻近设计区域、总体最大增幅与流量情景对应的数值结果；
- `outputs/figures/` 中的文件采用 `fig_<内容>.扩展名` 命名，并与 `main.tex` 中的图标签保持语义一致；
- `outputs/results.json` 记录主要设计、目标值、归一化结果和优化器参数。

题目附件和参考文献的版权归各自权利人所有，不包含在代码的 MIT License 许可范围内；论文引用信息保存在 `docs/paper/tex/references.bib` 中。

## 许可

代码按 [MIT License](LICENSE) 发布。赛题、附件和参考文献的版权归各自权利人所有，不包含在代码许可范围内。
