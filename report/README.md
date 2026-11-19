# B.Tech Major Project 1 - Academic Project Report (LaTeX)

**Project Title**: *Multi-Agent Deep Reinforcement Learning for Adaptive Traffic Signal Control in Indian Corridors*  
**Stage**: MAJOR PROJECT - I  
**Enrolment Numbers**: 23104003, 23104005, 23104030  
**Students**: Shivam Gupta, Tejas Srivastava, Anvit Singhal  
**Supervisor**: Dr. Kapil Madan, Assistant Professor  
**Department**: Department of Computer Science Engineering & Information Technology  
**Institution**: Jaypee Institute of Information Technology (JIIT), Noida  
**Date**: 20th November 2026  

---

## Repository Structure

```text
report/
|-- complete_report.tex          # [RECOMMENDED FOR OVERLEAF] Single standalone 1-file report (~60-70 pages)
|-- main.tex                     # Modular master file referencing chapters/
|-- README.md                    # Compilation guide and documentation
`-- chapters/
    |-- ch1_introduction.tex           # Chapter 1: Introduction, Objectives, Gantt, WBS, Risks
    |-- ch2_literature_survey.tex      # Chapter 2: Literature Survey, Taxonomy, Research Gaps
    |-- ch3_system_design.tex          # Chapter 3: System Design, Dec-POMDP Math, 4-Layer Architecture
    |-- ch4_implementation_testing.tex # Chapter 4: Implementation, Code, Test Cases, Benchmarks
    |-- ch5_results_discussion.tex     # Chapter 5: Empirical Results, Scenario Comparisons, BOM, SWOT
    |-- ch6_conclusion.tex             # Chapter 6: Conclusion, Societal/DPDP Impact, Future Work
    |-- references.tex                 # 22 IEEE Academic References (compact, 1.5 - 2 pages)
    |-- appendices.tex                 # Appendices A-E (Schemas, Math, Datasheets, Logs, Checklist)
    `-- annexures.tex                  # Annexures A-C (PO/PSO Mapping, Deliverables, UN SDGs)
```

---

## How to Compile

### Option 1: Overleaf (Fastest & 100% Guaranteed - 1 Click)
1. Open [Overleaf](https://www.overleaf.com) and log in.
2. Click **New Project** -> **Blank Project**.
3. Name your project (e.g., `JIIT_Major_Project_Report_Shivam_Tejas_Anvit`).
4. In Overleaf, open the default `main.tex`, delete its contents, and copy-paste the entire contents of [`report/complete_report.tex`](complete_report.tex).
5. Click **Recompile** (`Ctrl + Enter`).
   - *Note*: All diagrams (Architecture, Gantt chart, Delay comparisons, Power profile) are natively drawn using LaTeX `tikz` vectors, meaning **zero external image files are required to compile!**

### Option 2: Overleaf with Modular Chapters
1. Zip the entire `report` folder or drag-and-drop the files into Overleaf.
2. Ensure `main.tex` is selected as the main document.
3. Click **Recompile**.

### Option 3: Local LaTeX Compilation (TeX Live / MikTeX)
Run from terminal inside the `report` folder:
```bash
pdflatex main.tex
pdflatex main.tex
```
*(Running twice resolves Table of Contents, List of Figures, List of Tables, citations, and page references.)*

---

## Key Refinements Applied in this Update
1. **Zero Text Overflow**: All tables redesigned with `tabularx{\textwidth}` and proportional `X` columns so nothing spills past the page margins.
2. **Fixed Text Smushing & Underscores**: Replaced all unescaped underscores in file and node names with `\texttt{...}` with escaped `\_`, eliminating math-mode text merging.
3. **References Compacted to 1.5 - 2 Pages**: Standardized to `\begin{thebibliography}{99}` with 22 core IEEE citations, eliminating all `[?]` citation marks and keeping bibliography within 2 pages.
4. **Table 5.3 & TikZ Charts Cleaned**: Removed accidental delimiter collision in Table 5.3, properly spaced the legend in Figure 5.1, and removed unsupported Unicode emoji glyphs.
5. **Clean Annexure A Page Break**: Converted PO/PSO mapping to `longtable` so it breaks naturally across pages without colliding with the page footer.
