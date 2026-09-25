# Overleaf 使用说明

1. 将 `main.tex` 和 `references.bib` 上传到同一个 Overleaf 项目。
2. 在 **Menu → Compiler** 中选择 **XeLaTeX**。
3. 将所有“待定”“待完成”和红色写作提示替换为经团队核验、并由成员自行改写的内容。
4. 每个章节和小节标题中的主要作者必须替换为真实姓名。
5. 图片建议放在 `figures/`，用 `\includegraphics` 替换当前方框占位图。
6. 使用 `\citep{key}` 或 `\citet{key}` 引用 `references.bib` 中已核验的文献。
7. 提交前确认公开仓库 URL、全部定量结果、硬件信息、随机种子和模型标识。
8. 最终报告不得使用大学 Logo，也不要把报告 PDF 上传到公开代码仓库。

模板正文严格采用 `REPORT_OUTLINE.md` 中的七个课程必需章节，并保留其全部三级小节。附录用于模型清单和复现信息，不计入主体结构。

## 字体说明

模板已禁用 CTeX 默认的 Fandol fontset，并显式使用 Overleaf XeLaTeX 环境中的 `Noto Serif CJK SC` 和 `Noto Sans CJK SC`。不要使用 pdfLaTeX 或 LuaLaTeX 编译本模板。

如果项目仍报告找不到 Noto CJK，请把 Noto CJK 的 `.otf` 字体上传到项目，并在 `main.tex` 中将字体名称改成对应文件，例如 `\setCJKmainfont{NotoSerifCJKsc-Regular.otf}`。
