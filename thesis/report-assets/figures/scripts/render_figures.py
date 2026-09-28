#!/usr/bin/env python3
"""Render four report figures as vector PDFs from curated evidence CSVs."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data", HERE.parent / "output"
AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
CJK_FONT = next((font for font in ("Noto Sans CJK SC", "Source Han Sans CN", "Droid Sans Fallback")
                 if font in AVAILABLE_FONTS), "DejaVu Sans")
plt.rcParams.update({"font.size": 9, "font.family": ["DejaVu Sans", CJK_FONT],
                     "pdf.fonttype": 42, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def node(ax, xy, label, color="#e8f0fb", dashed=False, width=.15, height=.13):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
        facecolor=color, edgecolor="#26384f", linewidth=1.2, linestyle="--" if dashed else "-"))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8)

def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
        color="#26384f", linewidth=1.2, linestyle="--" if dashed else "-"))

def figure_01():
    fig, ax = plt.subplots(figsize=(10.5, 5.2)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("四阶段训练与模型选择流程", loc="left", weight="bold")
    ax.text(0, .94, "任务逐步增加导航、炸箱、对战与工程运行要求。", color="#4d5c70")
    labels = ["Task 1\n从零训练", "Task 2\n继续训练", "Task 3\n配对测试",
              "Task 4\n强对手", "打包与\n提交检查"]
    colors = ["#dcecff", "#e4f5e8", "#fff1c9", "#f9dce0", "#e9e2ff"]
    xs = [.01, .215, .42, .625, .83]
    for x, label, color in zip(xs, labels, colors): node(ax, (x, .60), label, color, width=.16)
    for x in xs[:-1]: arrow(ax, (x + .16, .665), (x + .205, .665))
    ax.text(.01, .30, "评估采用独立确认、配对比较、强对手测试与打包行为检查。", fontsize=8)
    ax.text(.01, .20, "来源 checkpoint 的历史基准和打包等价性检查分别描述比赛表现与封装行为。", fontsize=8)
    save(fig, "figure_01_curriculum_pipeline")

def figure_02():
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("模型推理与安全动作筛选流程", loc="left", weight="bold")
    ax.text(0, .93, "动作掩码排除不合适的动作，Q 网络对剩余动作排序。", color="#4d5c70")
    nodes = [(.01,.48,.11,"游戏状态"),(.17,.48,.15,"continuous-v2\n84 维特征"),(.37,.48,.13,"Double DQN\nQ 值"),
             (.56,.67,.15,"物理合法\n动作掩码"),(.56,.29,.15,"Survival Mask\n生存筛选"),
             (.76,.48,.16,"在剩余动作中\n选择最大 Q 值"),(.95,.48,.045,"动作")]
    for x,y,w,label in nodes: node(ax,(x,y),label,"#fff1c9" if x >= .56 else "#e8f0fb",width=w)
    arrow(ax,(.12,.545),(.17,.545)); arrow(ax,(.32,.545),(.37,.545)); arrow(ax,(.50,.545),(.56,.735))
    arrow(ax,(.50,.545),(.56,.355)); arrow(ax,(.71,.735),(.76,.545)); arrow(ax,(.71,.355),(.76,.545)); arrow(ax,(.92,.545),(.95,.545))
    ax.text(.01,.12,"合法动作和生存模块排除不满足条件的动作，Q 网络对剩余动作排序。",fontsize=8)
    ax.text(.01,.05,"打包检查验证重命名前后的行为一致性；比赛性能与安全性需要独立测试。",fontsize=8,color="#4d5c70")
    save(fig, "figure_02_inference_pipeline")

def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def figure_03():
    rows = read_csv("capability_progression.csv"); tasks = []
    for row in rows:
        if row["task"] not in tasks: tasks.append(row["task"])
    task_labels = {"Task 1 mean coins / 50": "Task 1：平均金币（满分 50）",
                   "Task 2 mean coins / 9": "Task 2：平均金币（满分 9）",
                   "Task 3 mean score": "Task 3：平均得分",
                   "Task 4 mean score": "Task 4：平均得分"}
    candidate_labels = {"Distilled CNN": "蒸馏 CNN", "Q(lambda) r20": "Q(lambda) r20",
                        "Distilled CNN D02": "蒸馏 CNN D02", "Parent seed22": "父模型 seed22",
                        "Child seed22/c150": "子模型 seed22/c150", "E1 seed11 endpoint": "E1 seed11 最终点",
                        "Die Hardest historical source checkpoint": "Die Hardest 来源 checkpoint 历史测试"}
    unit_labels = {"independent confirmation; n=100": "独立确认；n=100",
                   "frozen development; n=20": "开发测试；n=20",
                   "paired main validation; n=100": "配对主测试；n=100",
                   "specialist frozen endpoint; n=100": "专项模型最终点；n=100"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3)); fig.suptitle("各 Task 的代表性能力结果", x=.07, ha="left", weight="bold")
    fig.text(.07,.92,"各面板使用不同指标和测试设置，柱高仅表示对应面板内的结果。",color="#4d5c70")
    for ax, task in zip(axes.flat, tasks):
        rs = [r for r in rows if r["task"] == task]; labels=[candidate_labels.get(r["candidate"], r["candidate"]) for r in rs]; vals=[float(r["value"]) for r in rs]
        bars = ax.bar(range(len(vals)), vals, color="#4c78a8", edgecolor="#26384f", linewidth=.5)
        if task.startswith("Task 4"):
            bars[1].set_facecolor("#b9b9b9"); bars[1].set_hatch("///")
        ax.set_xticks(range(len(vals)), labels, rotation=15, ha="right", fontsize=7); ax.set_title(task_labels.get(task, task), loc="left", fontsize=10)
        ax.set_ylim(0,float(rs[0]["axis_max"])); ax.set_ylabel(unit_labels.get(rs[0]["unit"], rs[0]["unit"]), fontsize=8); ax.grid(axis="y",alpha=.25)
        for i,v in enumerate(vals): ax.text(i,v,f"{v:g}",ha="center",va="bottom",fontsize=7)
    fig.text(.07,.01,"Task 3 使用 100-world 配对主测试；Task 4 的 Die Hardest 来源 checkpoint 数值来自另一套历史 1,000-world 基准测试。",fontsize=8,color="#4d5c70")
    fig.tight_layout(rect=[.04,.05,1,.89]); save(fig,"figure_03_capability_progression")

def figure_04():
    rows=read_csv("task4_tradeoff.csv"); fig,ax=plt.subplots(figsize=(9.5,5.5))
    fig.suptitle("Task 4 得分、安全与推理延迟",x=.07,y=.98,ha="left",weight="bold")
    fig.text(.07,.92,"散点比较专项模型的 Task 4 得分与完整动作 P95 延迟。",color="#4d5c70")
    plot_rows = [r for r in rows if r["mean_score"] not in {"", "not_reported"}
                 and r["act_p95_ms"] not in {"", "not_reported"}]
    for r in plot_rows:
        x,y=float(r["act_p95_ms"]),float(r["mean_score"])
        ax.scatter(x,y,s=55,color="#4c78a8",edgecolor="#26384f")
        offsets={"E1 seed11 endpoint":(5,8),"E1 seed22 endpoint":(-60,15),"E1 seed33 endpoint":(5,5),
                 "Frozen C seed22":(5,5),"Frozen S seed22":(5,-13)}
        label = r["candidate"].replace(" endpoint", " 最终点")
        ax.annotate(label,(x,y),xytext=offsets.get(r["candidate"],(4,5)),textcoords="offset points",fontsize=7)
    ax.set_xlabel("完整 act 的 P95 延迟（ms）"); ax.set_ylabel("平均得分"); ax.grid(alpha=.25)
    fig.text(.07,.87,"图中专项实验均记录 0 自杀；安全轨迹在 step 77 复现反例 27155。",fontsize=7)
    fig.tight_layout(rect=[.03,.04,1,.84]); save(fig,"figure_04_task4_tradeoff")

if __name__ == "__main__":
    figure_01(); figure_02(); figure_03(); figure_04()
