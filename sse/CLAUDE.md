# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Marp-based slide deck workspace for a ~4-hour internal training on AI for robotics. The audience is lab students / new hires working on AI-enhanced articulated robots.

## Structure

Each topic lives in its own subdirectory (e.g., `machine_learning/`) with:
- `slides.md` — Marp slide source (the primary artifact)
- `fig/` — figures used in slides (SVG from draw.io, PNG)
- `script/` — Python scripts that generate figures
- `flowchart.md` — Mermaid diagrams (embedded in slides as needed)

## Commands

### Preview slides (Marp CLI)
```bash
marp --watch machine_learning/slides.md
```

### Export to HTML/PDF
```bash
marp machine_learning/slides.md -o machine_learning/slides.html
marp machine_learning/slides.pdf --pdf machine_learning/slides.pdf
```

### Generate figures
```bash
cd machine_learning/script
python logit.py
python sigmoid.py
python dimensional_reduction.py
```
Figures are saved relative to the script (`../fig/`), so run scripts from within `script/`.

## Slide authoring conventions

- All slides use Marp with KaTeX math (`math: katex`), 16:9 ratio, and a shared CSS header defined at the top of each `slides.md`.
- `h1` is positioned absolutely (top bar style) — keep slide titles as a single `#` heading.
- Math equations use `$$...$$` display blocks or `$...$` inline.
- Figures are referenced as relative paths from `slides.md`, e.g., `![](fig/foo.png)`.

## Content guidelines (from parent CLAUDE.md)

- 技術の正確性を最重要視。ごまかしのない説明。
- AI専門用語は省略せず、注釈を入れて正面から説明する。
- 例え話・絵文字・歴史的背景は不要。
- コードサンプルは Python / PyTorch を使用する。
