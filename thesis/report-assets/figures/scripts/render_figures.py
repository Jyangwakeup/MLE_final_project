#!/usr/bin/env python3
"""Render the four report figures from curated real-evidence CSV data."""
from __future__ import annotations
import csv
import html
import shutil
import subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
DATA, OUT = FIGURES / "data", FIGURES / "output"

def wrap(parts):
    return "\n".join(['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="900" viewBox="0 0 1500 900">', '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:30px;font-weight:bold}.label{font-size:18px}.small{font-size:14px}.note{font-size:13px;fill:#4d5c70}.axis{stroke:#45576d;stroke-width:2}.grid{stroke:#d5dce5;stroke-width:1}</style>', '<rect width="1500" height="900" fill="white"/>', *parts, '</svg>'])

def write(name, parts):
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f".{name}.tmp.svg"
    pdf_path = OUT / f"{name}.pdf"
    svg_path.write_text(wrap(parts), encoding="utf-8")
    inkscape = shutil.which("inkscape")
    if not inkscape:
        svg_path.unlink()
        raise RuntimeError("Inkscape is required to render report figures as PDF")
    try:
        subprocess.run(
            [inkscape, str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"],
            check=True,
        )
    finally:
        svg_path.unlink(missing_ok=True)

def box(x, y, w, h, label, fill="#e8f0fb", dashed=False):
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#31445f" stroke-width="2"{dash}/>', f'<text x="{x+w/2}" y="{y+h/2-8}" text-anchor="middle" class="label">{html.escape(label)}</text>']

def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    return f'<path d="M {x1} {y1} L {x2} {y2}" stroke="#31445f" stroke-width="3" fill="none" marker-end="url(#arrow)"{dash}/>'
MARKER = '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#31445f"/></marker></defs>'

def figure_01():
    p = [MARKER, '<text x="70" y="70" class="title">Figure 1. Curriculum, frozen selection, and package verification</text>', '<text x="70" y="100" class="note">Solid: recorded evidence. Dashed: planned or incomplete stage. Model families remain separate in Tasks 1 and 2.</text>']
    xs = [80, 360, 640, 920, 1200]
    for x, label, fill in zip(xs, ["Task 1 — Cold start", "Task 2 — Warm start", "Task 3 — Paired validation", "Task 4 — Strong opponents", "Package / submission check"], ["#dcecff", "#e4f5e8", "#fff1c9", "#f9dce0", "#e9e2ff"]): p += box(x, 250, 210, 110, label, fill)
    for x in xs[:-1]: p.append(arrow(x + 210, 305, x + 280, 305))
    p += box(360, 530, 210, 110, "Development checkpoint selection", "#fff", True)
    p += box(640, 530, 210, 110, "Confirmation / main validation", "#fff", True)
    p += [arrow(465, 360, 465, 530, True), arrow(745, 360, 745, 530, True), '<text x="80" y="740" class="label">Observed evidence:</text>', '<text x="110" y="780" class="small">Q-learning and distilled CNN: Task 1 independent confirmations recorded.</text>', '<text x="110" y="810" class="small">Task 3 seed22/c150: 100-world paired main validation. Task 4: bounded attempts and historical B33 benchmark are separate protocols.</text>', '<text x="110" y="850" class="note">No line asserts every planned Task 4 stage or reserved final-test set was run.</text>']
    write("figure_01_curriculum_pipeline", p)

def figure_02():
    p = [MARKER, '<text x="70" y="70" class="title">Figure 2. Inference and safety-action admission pipeline</text>', '<text x="70" y="100" class="note">Safety masks filter inadmissible actions; they do not rank a preferred action.</text>']
    nodes = [(70,330,180,100,"Game state"),(300,330,200,100,"continuous-v2 — 84 features"),(550,330,190,100,"Double DQN — Q values"),(790,230,210,100,"physical legal action mask"),(790,450,210,100,"survival admission mask"),(1050,330,220,100,"argmax over admissible actions"),(1320,330,120,100,"Action")]
    for x,y,w,h,label in nodes: p += box(x,y,w,h,label,"#e8f0fb" if x < 790 else "#fff1c9")
    p += [arrow(250,380,300,380),arrow(500,380,550,380),arrow(740,380,790,280),arrow(740,380,790,500),arrow(1000,280,1050,380),arrow(1000,500,1050,380),arrow(1270,380,1320,380),'<text x="70" y="700" class="label">Boundary:</text>','<text x="170" y="700" class="small">The learner ranks Q values; legality and survival modules veto only actions outside their admissible sets.</text>','<text x="170" y="735" class="note">Package validation confirms source/renamed behavioural equivalence, not a new competition evaluation or complete safety proof.</text>']
    write("figure_02_inference_pipeline", p)

def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def bars(p, x, y, width, height, title, entries, maximum, unit):
    p += [f'<text x="{x}" y="{y-25}" class="label">{html.escape(title)}</text>', f'<line x1="{x}" y1="{y+height}" x2="{x+width}" y2="{y+height}" class="axis"/>']
    for n in range(5):
        gy = y + height - n * height / 4
        p += [f'<line x1="{x}" y1="{gy}" x2="{x+width}" y2="{gy}" class="grid"/>', f'<text x="{x-8}" y="{gy+5}" text-anchor="end" class="small">{maximum*n/4:.0f}</text>']
    gap = width / len(entries)
    for i, (label, value) in enumerate(entries):
        bh, bx = value / maximum * height, x + i * gap + 20
        p += [f'<rect x="{bx}" y="{y+height-bh}" width="{gap-40}" height="{bh}" fill="#4c78a8"/>', f'<text x="{bx+(gap-40)/2}" y="{y+height-bh-8}" text-anchor="middle" class="small">{value:.2f}</text>', f'<text x="{bx+(gap-40)/2}" y="{y+height+28}" text-anchor="middle" class="small">{html.escape(label)}</text>']
    p.append(f'<text x="{x+width}" y="{y-25}" text-anchor="end" class="note">{html.escape(unit)}</text>')

def figure_03():
    rows = read_csv("capability_progression.csv")
    by_task = {}
    for row in rows: by_task.setdefault(row["task"], []).append((row["candidate"], float(row["value"])))
    p = ['<text x="70" y="55" class="title">Figure 3. Frozen capability evidence by task</text>', '<text x="70" y="85" class="note">Panels use different metrics and protocols; bars are not a continuous learning curve or cross-panel effect-size comparison.</text>']
    for (task, entries), (x,y) in zip(by_task.items(), [(90,180),(820,180),(90,570),(820,570)]):
        row = next(r for r in rows if r["task"] == task)
        bars(p, x, y, 560, 190, task, entries, float(row["axis_max"]), row["unit"])
    p.append('<text x="70" y="875" class="note">Task 3 uses parent/child scores from a paired 100-world main validation; Task 4 B33 is a separate historical 1,000-world benchmark.</text>')
    write("figure_03_capability_progression", p)

def figure_04():
    rows = read_csv("task4_tradeoff.csv")
    p = ['<text x="70" y="60" class="title">Figure 4. Task 4 score, safety, and latency records</text>', '<text x="70" y="90" class="note">Circles: frozen specialist evaluations. B33 historical benchmark has no P95; package check has no score. Neither is plotted as a point.</text>', '<line x1="150" y1="760" x2="1350" y2="760" class="axis"/><line x1="150" y1="150" x2="150" y2="760" class="axis"/>']
    for n in range(6):
        xx, yy = 150+n*240, 760-n*110
        p += [f'<line x1="{xx}" y1="150" x2="{xx}" y2="760" class="grid"/><text x="{xx}" y="790" text-anchor="middle" class="small">{n*5} ms P95</text>', f'<line x1="150" y1="{yy}" x2="1350" y2="{yy}" class="grid"/><text x="140" y="{yy+5}" text-anchor="end" class="small">{n}</text>']
    p += ['<text x="700" y="835" text-anchor="middle" class="label">act P95 latency (ms)</text>','<text x="48" y="480" transform="rotate(-90 48 480)" text-anchor="middle" class="label">mean score</text>']
    label_offsets = {
        "E1 seed11 endpoint": (-110, -32, "E1-11"),
        "E1 seed22 endpoint": (15, -50, "E1-22"),
        "E1 seed33 endpoint": (15, 24, "E1-33"),
        "Frozen C seed22": (-115, 22, "C-22"),
        "Frozen S seed22": (15, 45, "S-22"),
        "B33 historical 1000-world": (18, -12, "B33 historical"),
    }
    for r in rows:
        if r["mean_score"] == "not_reported":
            p.append(f'<rect x="70" y="852" width="18" height="18" fill="#999"/><text x="100" y="866" class="small">{html.escape(r["candidate"])}: score not measured</text>')
            continue
        if r["act_p95_ms"] == "not_reported":
            p.append('<text x="760" y="866" class="small">B33 historical: score 4.231, P95 not reported</text>')
            continue
        x, y = 150+float(r["act_p95_ms"])/25*1200, 760-float(r["mean_score"])/5*550
        color = "#e45756" if r["record_type"] == "historical_benchmark" else "#4c78a8"
        mark = f'<polygon points="{x},{y-13} {x+13},{y} {x},{y+13} {x-13},{y}" fill="{color}"/>' if r["record_type"] == "historical_benchmark" else f'<circle cx="{x}" cy="{y}" r="11" fill="{color}"/>'
        dx, dy, label = label_offsets[r["candidate"]]
        p += [mark, f'<text x="{x+dx}" y="{y+dy}" class="small">{label}</text>']
    p.append('<text x="160" y="125" class="note">Safety note: plotted evaluated attempts report 0 observed suicide; B33 retains an unresolved historical safety-contract counterexample.</text>')
    write("figure_04_task4_tradeoff", p)

if __name__ == "__main__":
    figure_01(); figure_02(); figure_03(); figure_04()
