# Thesis 工作区导航

## 编译入口

- [`main_zh.tex`](main_zh.tex)：当前中文稿。
- [`main.tex`](main.tex)：英文提交入口。
- [`references.bib`](references.bib)：论文参考文献数据库。
- [`report-assets/`](report-assets/)：共享表格、图表数据、绘图脚本和论文使用的图表 PDF。LaTeX 资源路径依赖此目录。

## 支持文档

- [`docs/planning/`](docs/planning/)：报告结构大纲和只读模板。
- [`docs/research/`](docs/research/)：跨路线实验分析、证据故事和模型实验汇总。
- [`docs/writing/`](docs/writing/)：写作要求、写作风格指南和 Overleaf 使用说明。
- [`members/`](members/)：按主要资料负责人整理的路线记录和来源索引。共享组件、团队结论和汇总数据仍以公共文档及 `report-assets/` 为准。

## 本地编译

`main_zh.tex` 保留在 `thesis/` 根目录，便于 Overleaf 直接将它设为中文主文档。VS Code 的 LaTeX Workshop 已配置为使用 XeLaTeX 和 `latexmk`，并将 PDF、日志及辅助文件输出到 `thesis/build/`。

也可以在 `thesis/` 目录手动运行：

```sh
latexmk -xelatex -outdir=build main_zh.tex
```

`build/` 被 Git 忽略。论文图片仍从 `report-assets/figures/output/` 读取。Overleaf 入口和设置见 [`docs/writing/OVERLEAF_README.md`](docs/writing/OVERLEAF_README.md)。
