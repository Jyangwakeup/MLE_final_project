# Overleaf 使用说明

1. 上传整个外层 `thesis/` 目录的内容，不要只上传原来的内层源码目录。上传后的 Overleaf 项目根目录必须直接包含 `main.tex`、`references.bib` 和 `report-assets/`。
2. 在 **Menu → Main document** 中选择项目根目录的 `main.tex`。
3. 在 **Menu → Compiler** 中选择 **XeLaTeX**；不要使用 pdfLaTeX 或 LuaLaTeX。
4. 保持图片路径为 `report-assets/figures/output/*.pdf`。论文图片统一使用由 SVG 源文件导出的矢量 PDF；如果没有上传 `report-assets/`，文档会因缺图而停止编译。
5. 使用 `\citep{key}` 或 `\citet{key}` 引用根目录 `references.bib` 中已核验的文献。
6. 将所有“待定”“待完成”和红色写作提示替换为经团队核验、并由成员自行改写的内容。
7. 每个章节和小节标题中的主要作者必须替换为真实姓名。
8. 提交前确认公开仓库 URL、全部定量结果、硬件信息、随机种子和模型标识。
9. 最终报告不得使用大学 Logo，也不要把报告 PDF 上传到公开代码仓库。

正确的 Overleaf 根目录结构应为：

```text
main.tex
references.bib
report-assets/
  figures/output/*.pdf
  tables/*.csv
OVERLEAF_README.md
```

本地遗留的 `thesis/build/` 仅包含旧辅助文件，不需要上传，也不要把其中任何文件设置为 Main document。

模板正文严格采用 `REPORT_OUTLINE.md` 中的七个课程必需章节，并保留其全部三级小节。附录用于模型清单和复现信息，不计入主体结构。

## 字体说明

模板已禁用 CTeX 默认的 Fandol fontset，并显式使用 Overleaf XeLaTeX 环境中的 `Noto Serif CJK SC` 和 `Noto Sans CJK SC`。不要使用 pdfLaTeX 或 LuaLaTeX 编译本模板。

如果项目仍报告找不到 Noto CJK，请把 Noto CJK 的 `.otf` 字体上传到项目，并在 `main.tex` 中将字体名称改成对应文件，例如 `\setCJKmainfont{NotoSerifCJKsc-Regular.otf}`。
