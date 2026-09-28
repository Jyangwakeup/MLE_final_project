# Overleaf 使用说明

> 写作顺序：`main_zh.tex` 完整中文稿 → 内容冻结与主笔分配 → `main_en.tex` 英文审校稿 → `main.tex` 最终英文提交入口。三个证据输入包已经满足中文写作入口，当前处于中文初稿团队核验阶段。

1. 上传论文需要的 TeX 文件、`references.bib` 和 `report-assets/`。`main_zh.tex`、`main.tex` 与 `references.bib` 保持在项目根目录，`report-assets/` 也保留原路径；`docs/` 和 `members/` 是支持材料，不是编译依赖。
2. 在中文写作与事实核验阶段，将 **Menu → Main document** 设为 `main_zh.tex`。中文冻结并创建英文稿后，改为 `main_en.tex`。只有最终提交检查时才选择 `main.tex`。
3. 在 **Menu → Compiler** 中选择 **XeLaTeX**；不要使用 pdfLaTeX 或 LuaLaTeX。
4. 保持图片路径为 `report-assets/figures/output/*.pdf`。论文图片由 Matplotlib 直接输出矢量 PDF；如果没有上传 `report-assets/`，文档会因缺图而停止编译。
5. 使用 `\citep{key}` 或 `\citet{key}` 引用根目录 `references.bib` 中已核验的文献。
6. 中文内容冻结前，将所有正文 TODO 和蓝色写作提示替换为经团队核验的完整内容；主笔栏暂时保留为空白占位。
7. 中文内容冻结后、英文翻译前，为每个章节和小节填写真实主要作者，并由对应主笔实质性复核。
8. 提交前确认公开仓库 URL、全部定量结果、硬件信息、随机种子和模型标识。2026-09-27 的匿名访问检查对当前 GitHub URL 返回 404，必须在提交前公开仓库并重新进行外部访问检查。
9. 最终报告不得使用大学 Logo，也不要把报告 PDF 上传到公开代码仓库。

最终提交阶段的 Overleaf 根目录结构应为：

```text
main.tex               # 最终英文提交入口，与审核通过的 main_en.tex 完全一致
main_en.tex            # 中文冻结且分配主笔后创建
main_zh.tex            # 中文事实基线
references.bib
report-assets/
  figures/output/*.pdf
  tables/*.csv
docs/writing/OVERLEAF_README.md  # 使用说明，可不上传
```

本地运行 `latexmk -xelatex -outdir=build main_zh.tex` 时，辅助文件、日志和 PDF 会写入 `build/`。该目录仅供本地编译并被 Git 忽略，不需要上传，也不要把其中任何文件设置为 Main document。

模板正文严格采用 [`docs/planning/REPORT_OUTLINE.md`](../planning/REPORT_OUTLINE.md) 中的七个课程必需章节，并保留其全部三级小节。附录用于模型清单和复现信息，不计入主体结构。

## 阶段切换规则

- 三个完整证据输入包未齐备时，只整理证据，不填充完整论文正文；本项目已满足此门槛。
- 中文初稿按研究问题统一整合三条路线，除封面团队署名外不写个人贡献归属。
- 中文内容冻结后填写主笔栏和 Project Planning 贡献表，再创建 `main_en.tex`。
- 事实或结构修订同步到中英文；仅英文表达层面的润色不回写中文。
- 提交前将 `main_en.tex` 完整同步为 `main.tex`，并确认二者内容一致。

## 字体说明

中文稿已禁用 CTeX 默认的 Fandol fontset，并优先使用 Overleaf XeLaTeX 环境中的 `Noto Serif CJK SC` 和 `Noto Sans CJK SC`。本地缺少 Noto CJK 时，`main_zh.tex` 会尝试使用 `Droid Sans Fallback`。不要使用 pdfLaTeX 或 LuaLaTeX 编译中文稿。

如果项目仍报告找不到 Noto CJK，请把 Noto CJK 的 `.otf` 字体上传到项目，并在 `main_zh.tex` 中将字体名称改成对应文件，例如 `\setCJKmainfont{NotoSerifCJKsc-Regular.otf}`。
