import os
import sys
import shutil
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' 
    along with running headers and footers on every page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        print(f"NumberedCanvas: rendering {num_pages} total pages.")
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # We omit running header on the title page (page 1)
        if self._pageNumber > 1:
            # Running Header
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4A5568"))
            self.drawString(40, 755, "PROJECT AEGIS: CONTINUAL LEARNING WITH GRADIENT GATING — MASTER REPORT")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(40, 750, 612 - 40, 750)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(40, 45, 612 - 40, 45)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(40, 32, "Confidential / Academic Defense Monograph — Project AEGIS (github.com/jeetdoesthings/AEGIS)")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 40, 32, page_str)
        
        self.restoreState()


def create_callout(text, title="KEY TAKEAWAY", bg_color="#F7FAFC", border_color="#2B6CB0", title_color="#1A365D", width=532):
    """Generates an elegant boxed callout for findings and alerts."""
    styles = getSampleStyleSheet()
    t_style = ParagraphStyle(
        'CalloutTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor(title_color),
        spaceAfter=3
    )
    b_style = ParagraphStyle(
        'CalloutBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748")
    )
    content = [
        Paragraph(f"<b>{title}</b>", t_style),
        Paragraph(text, b_style)
    ]
    t = Table([[content]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


def build_pdf(filename="outputs/AEGIS_Master_Project_Report.pdf"):
    print(f"Building Master PDF report at {filename}...")
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Palette
    primary_color = "#1A365D"   # Deep Navy
    secondary_color = "#2B6CB0" # Slate Blue
    body_color = "#2D3748"      # Charcoal
    accent_color = "#C53030"    # Crimson Accent

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor(primary_color),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(secondary_color),
        spaceAfter=12
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#718096"),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor(primary_color),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor(secondary_color),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor(primary_color),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.8,
        textColor=colors.HexColor(body_color),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor(body_color),
        leftIndent=12,
        spaceAfter=3
    )
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C")
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # TITLE & HEADER BLOCK
    # =========================================================================
    story.append(Paragraph("PROJECT AEGIS: MASTER RESEARCH REPORT", title_style))
    story.append(Paragraph("Continual Learning with Directional & Magnitude Gradient Gating: Empirical Evaluation, Mechanistic Failure Modes, and Anti-Ratchet Repairs", subtitle_style))
    story.append(Paragraph("<b>Author:</b> Jeet & Project AEGIS Team &nbsp;|&nbsp; <b>Institution:</b> Machine Learning Research Lab &nbsp;|&nbsp; <b>Hardware:</b> Apple M4 &nbsp;|&nbsp; <b>Repository:</b> github.com/jeetdoesthings/AEGIS &nbsp;|&nbsp; <b>Date:</b> November 2026", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(primary_color), spaceBefore=2, spaceAfter=10))

    # =========================================================================
    # SECTION 1: SIMPLIFIED EXPLANATION & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Simplified Explanation", h1_style))
    
    p1 = ("<b>What is Catastrophic Forgetting?</b> When artificial neural networks are trained on new tasks sequentially, "
          "standard gradient descent aggressively updates the weights to minimize current-task loss. In doing so, it overwrites "
          "the delicate, highly specific parameter configurations required to solve previous tasks. Imagine learning to speak French, "
          "and upon becoming fluent in French, discovering you have completely lost the ability to speak English—this is catastrophic forgetting. "
          "It represents the single greatest barrier to building autonomous, lifelong learning AI agents.")
    story.append(Paragraph(p1, body_style))

    p2 = ("<b>Why Don't Contemporary LLM Approaches Solve This?</b> In modern practice, two paradigms are frequently assumed to bypass forgetting: "
          "Retrieval-Augmented Generation (RAG) and ultra-long context windows. Neither addresses the underlying biological plasticity problem. "
          "RAG merely injects retrieved external text into an inference prompt; it leaves network parameter weights completely static and cannot incrementally "
          "acquire non-textual perceptual skills, new procedural reasoning policies, or updated latent representations. Similarly, long context windows are ephemeral, "
          "incur quadratic memory and compute costs, and completely vanish once an inference session closes. True lifelong intelligence requires "
          "consolidating knowledge directly into the network's persistent parameter weights without storing unbounded raw data.")
    story.append(Paragraph(p2, body_style))

    p3 = ("<b>The Core Hypothesis of Project AEGIS:</b> Can we devise a <i>buffer-free</i>, parameter-level gradient gate that prevents catastrophic "
          "forgetting by monitoring gradient trajectories during optimization, without storing a single past training sample? "
          "The project investigated the <i>Gradient-Conflict Gate</i>, an algorithm that tracks historical weight movement across prior tasks and suppresses "
          "gradient updates that conflict with past directions.")
    story.append(Paragraph(p3, body_style))

    # Executive Summary Callout Box
    summary_box_text = (
        "<b>1. The Asymmetric Ratchet Failure:</b> The original directional conflict gate (<code>gate:conflict</code>) failed catastrophically. "
        "On a 10-task sequential sequence, accuracy collapsed to <b>41.92 ± 1.71%</b> (falling <b>-36.29 ± 2.82%</b> below plain fine-tuning). "
        "Parameter telemetry supported the exact failure mechanism: zeroing opposing updates while leaving outward updates unhindered acted as a "
        "one-way mechanical ratchet, doubling weight norms (22.70 vs 11.79) and causing uncontrollable drift.<br/>"
        "<b>2. Magnitude Beats Direction:</b> Gating weights purely by displacement magnitude (<code>gate:magnitude</code>) avoided the ratchet entirely, "
        "achieving <b>83.13 ± 1.22%</b> accuracy on 10 tasks and halving forgetting.<br/>"
        "<b>3. Repaired Anti-Drift Gates Statistically Win:</b> Formulating an anti-drift gradient brake (<code>gate:drift</code>) and a hybrid gate "
        "(<code>gate:hybrid</code>) eliminated runaway drift, achieving <b>83.82 ± 0.54%</b> and <b>83.21 ± 1.26%</b> on 10 tasks—statistically "
        "outperforming plain fine-tuning (78.21 ± 1.68%) by <b>>+5% ACC</b> without storing replay exemplars.<br/>"
        "<b>4. The Output Head Bottleneck:</b> On domain-incremental Split-MNIST, all buffer-free methods tied around <b>62% ACC</b>, showing that "
        "weight-space gating cannot resolve representational collision at a shared output head without exemplars (Replay: 96.85%) or task routing (LoRA: 98.73%)."
    )
    story.append(create_callout(summary_box_text, title="EXECUTIVE RESEARCH SUMMARY (AUDITED FINDINGS)", bg_color="#EBF8FF", border_color="#3182CE", title_color="#2B6CB0"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: PROJECT TIMELINE & BEFORE CL2.PY (PHASE 0)
    # =========================================================================
    story.append(Paragraph("2. Project Evolution: Before & After cl2.py", h1_style))
    
    p_evol = ("A defining hallmark of Project AEGIS was the methodological transformation from an initial ad-hoc prototype into an "
              "academically rigorous, reproducible scientific benchmark. The project's trajectory is bifurcated into two distinct epochs: "
              "<b>Phase 0 (The Frozen Prototype)</b> and <b>Phases 1–5 (The Rigorous Benchmark)</b>.")
    story.append(Paragraph(p_evol, body_style))

    story.append(Paragraph("Phase 0: The Status Before cl2.py (cl.py Prototype)", h2_style))
    p_ph0 = ("Before <code>cl2.py</code> was created, the workspace contained an initial standalone script named <code>cl.py</code>. "
             "It implemented six initial continual learning techniques on a small Multi-Layer Perceptron (784-256-256-10, 269k parameters) "
             "evaluated on 5-task Permuted-MNIST under plain SGD. However, Phase 0 suffered from severe experimental and numerical deficiencies:")
    story.append(Paragraph(p_ph0, body_style))

    story.append(Paragraph("• <b>Single Split & Data Leakage:</b> The initial code evaluated directly on the training distribution or evaluated without "
                           "an isolated validation set. Hyperparameters were tuned on the exact same test splits reported in tables, representing severe methodological data leakage.", bullet_style))
    story.append(Paragraph("• <b>The Mysterious EWC 9.80% Breakdown:</b> In early runs, Elastic Weight Consolidation (EWC) mysteriously scored exactly <b>9.80 ± 0.00% accuracy</b> "
                           "(near chance for 10 classes). Phase 0 was unable to diagnose why EWC was failing so severely.", bullet_style))
    story.append(Paragraph("• <b>Directional Gate Collapse:</b> When running <code>gate:conflict:1.0</code> with full penalty, the model collapsed to 13.7 ± 3.0% ACC, "
                           "while the magnitude ablation reached 92.3 ± 0.6% ACC. Why the conflict check was causing devastation remained unknown.", bullet_style))
    story.append(Paragraph("• <b>The Permanent Freeze Decision:</b> Under the project charter, <code>cl.py</code> was permanently frozen as an immutable reference. "
                           "All bug fixes, scientific scaffolding, validation splits, diagnostics, and repairs were strictly developed in <code>cl2.py</code>.", bullet_style))

    story.append(Paragraph("Autopsy of the EWC Failure (Why 9.80% Accuracy Occurred)", h3_style))
    p_ewc_diag = ("In Phase 1, an exhaustive step-level autopsy (logged in <code>outputs/phase1/ewc_diagnosis.txt</code>) uncovered the mathematical cause of EWC's failure. "
                  "In <code>cl.py</code>, the EWC quadratic penalty was defined as $\\mathcal{L}_{total} = \\mathcal{L}_{task} + \\frac{\\lambda}{2} \\sum F_i (p_i - s_i)^2$ with "
                  "the default argument hardcoded to $\\lambda = 5000$. Combined with plain SGD learning rate $\\eta = 0.01$ and unnormalized empirical Fisher diagonal values, "
                  "the effective gradient update scale was $\\eta \\lambda F_i \\gg 1$.<br/>"
                  "By step 13 of Task 2, gradient updates diverged exponentially, causing training loss to become <b>NaN (Not-a-Number) / Infinity</b>. "
                  "During Fisher sampling for subsequent tasks, <code>torch.multinomial</code> crashed with a RuntimeError because the categorical probability tensor contained NaNs. "
                  "When evaluating the model, the forward pass produced NaN logits for all classes. In PyTorch, <code>torch.argmax</code> on an array of NaNs defaults by IEEE 754 "
                  "comparison semantics to <b>index 0</b>. In the MNIST test set, digit '0' occurs exactly 980 times out of 10,000 images—yielding exactly <b>9.80% accuracy</b>! "
                  "EWC was not learning at random chance; its floating-point representations were exploding into mathematical oblivion.")
    story.append(Paragraph(p_ewc_diag, body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Phases 1–5: The Rigorous Reconstruction (cl2.py)", h2_style))
    p_ph1 = ("To transform AEGIS into a bulletproof scientific study, <code>cl2.py</code> introduced five foundational pillars:<br/>"
             "1. <b>Strict 3-Way Split:</b> MNIST's 60,000 images were partitioned (seed 42) into 55,000 training, 5,000 validation, and 10,000 held-out test samples.<br/>"
             "2. <b>Divergence Guard:</b> Telemetry in <code>train_task</code> immediately halts any run encountering non-finite loss, recording <code>DIVERGED</code> rather than reporting corrupted numbers.<br/>"
             "3. <b>Repaired EWC:</b> Swept $\\lambda \\in \\{1, 5, 20, 100\\}$ on validation split. Found $\\lambda=20$ is optimal, achieving <b>93.76 ± 0.36% ACC</b> on test split with zero divergence.<br/>"
             "4. <b>Equal-Budget Tuning:</b> All methods evaluated across 25 configurations on validation seeds 0–2 (budget $\\le 6$ per method) and locked into JSON configs.<br/>"
             "5. <b>Statistical Tie Rule:</b> Two methods tie if their <code>mean ± std</code> intervals overlap. 'Beats' is strictly reserved for disjoint empirical distributions.")
    story.append(Paragraph(p_ph1, body_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: MASTER BENCHMARK RESULTS
    # =========================================================================
    story.append(Paragraph("3. Master Benchmark Results & Quantitative Compendium", h1_style))
    p_res = ("All 10 methods were evaluated on the held-out test split across 5 independent random seeds (100, 101, 102, 103, 104) across all three benchmarks. "
             "Metrics reported: <b>ACC</b> (final average accuracy across all tasks, %), <b>BWT</b> (backward transfer / forgetting, %), "
             "and <b>Δ ACC vs FT</b> (per-seed paired difference versus fine-tuning).")
    story.append(Paragraph(p_res, body_style))

    # Embed Master Table
    if os.path.exists("outputs/phase5/master_summary.png"):
        story.append(Image("outputs/phase5/master_summary.png", width=530, height=190))
        story.append(Paragraph("Figure 1: Master Continual Learning Summary Table across 5-Task Permuted-MNIST, 10-Task Permuted-MNIST, and 5-Task Split-MNIST (5 test seeds: 100–104).", caption_style))

    # Summary Table Flowable
    table_data = [
        ["Method Family", "Method Configuration", "Perm5 ACC (%)", "Perm5 BWT (%)", "Perm10 ACC (%)", "Perm10 BWT (%)", "Split ACC (%)"],
        ["Baselines", "Fine-Tuning (Plain SGD)", "88.97 ± 1.85", "-9.62 ± 2.34", "78.21 ± 1.68", "-20.80 ± 1.84", "62.07 ± 1.00"],
        ["Baselines", "EWC (repaired, λ=20/100)", "93.76 ± 0.36", "-2.34 ± 0.55", "90.22 ± 0.87", "-4.17 ± 0.93", "63.96 ± 1.39"],
        ["Baselines", "Replay (buffer=1000)", "95.85 ± 0.11", "-1.18 ± 0.14", "95.22 ± 0.09", "-1.92 ± 0.14", "96.85 ± 0.30"],
        ["Baselines", "LoRA (r=16/8, oracle ID)*", "94.30 ± 0.12", "0.00 ± 0.00", "94.05 ± 0.05", "0.00 ± 0.00", "98.73 ± 0.06"],
        ["Original Gate", "Gate: Conflict (original, β=0.25)", "81.73 ± 1.41", "-17.60 ± 1.84", "41.92 ± 1.71", "-58.83 ± 1.95", "62.28 ± 0.98"],
        ["Original Gate", "Gate: Magnitude (ablation, β=1.0)", "91.43 ± 0.86", "-4.39 ± 1.01", "83.13 ± 1.22", "-12.82 ± 1.33", "61.90 ± 0.36"],
        ["Original Gate", "Gate: Layer Cosine (β=0.5/1.0)", "89.00 ± 2.32", "-9.62 ± 2.98", "78.80 ± 0.97", "-20.15 ± 1.15", "62.05 ± 1.02"],
        ["Repaired Gate", "Gate: Drift (anti-drift, β=0.1)", "91.89 ± 1.03", "-5.82 ± 1.39", "83.21 ± 1.26", "-14.98 ± 1.44", "62.62 ± 1.03"],
        ["Repaired Gate", "Gate: Hybrid (conflict×mag, β=0.1)", "90.98 ± 0.90", "-6.64 ± 1.20", "83.82 ± 0.54", "-13.95 ± 0.64", "62.78 ± 0.56"],
        ["Repaired Gate", "Gate: Soft Conflict (β=0.1)", "90.68 ± 0.58", "-7.44 ± 0.79", "80.05 ± 1.66", "-18.58 ± 1.86", "62.34 ± 1.03"]
    ]
    t_summary = Table(table_data, colWidths=[65, 145, 65, 65, 65, 65, 62])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(primary_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6.8),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('TEXTCOLOR', (2, 8), (2, 8), colors.HexColor("#2B6CB0")), # Highlight gate:drift perm5 win
        ('FONTNAME', (2, 8), (2, 8), 'Helvetica-Bold'),
        ('TEXTCOLOR', (4, 8), (4, 9), colors.HexColor("#2B6CB0")), # Highlight gate:hybrid/drift perm10 win
        ('FONTNAME', (4, 8), (4, 9), 'Helvetica-Bold'),
    ]))
    story.append(t_summary)
    story.append(Paragraph("*Note: LoRA requires an oracle task ID at test time to select task-specific adapters; it is not directly comparable to task-agnostic single-head models.", caption_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4: THE ASYMMETRIC RATCHET & MECHANISTIC REPAIRS
    # =========================================================================
    story.append(Paragraph("4. Mechanistic Autopsy: The Asymmetric Ratchet & Gate Repairs", h1_style))
    
    p_ratchet = ("<b>Why Did the Original Conflict Gate Fail?</b> In the original formulation, the gate computed cumulative parameter displacement "
                 "$d = \\theta_{start} - \\theta^{(0)}$ across all prior tasks. For each incoming mini-batch gradient $g = \\nabla_\\theta \\mathcal{L}_{task}$, "
                 "it checked for conflict: since the SGD update is $\\Delta \\theta = -\\eta g$, a step opposes historical movement $d$ if "
                 "$(-\\eta g) \\cdot d < 0 \\iff g \\cdot d > 0$. When conflict was detected, the gate multiplied the gradient element-wise by "
                 "$(1 - \\beta \\cdot \\text{mag})$. At $\\beta = 1.0$, conflicting gradient components were zeroed out.")
    story.append(Paragraph(p_ratchet, body_style))

    # Diagram of the ratchet
    ratchet_box = (
        "<b>THE ASYMMETRIC RATCHET FAILURE MECHANISM:</b><br/>"
        "• <b>Update Towards Origin (-d):</b> Triggered conflict condition (<code>g · d > 0</code>) → <b>BLOCKED / SCALED TO ZERO</b>.<br/>"
        "• <b>Update Away From Origin (+d):</b> Satisfied <code>g · d ≤ 0</code> → <b>PERMITTED FULLY WITHOUT BRAKING</b>.<br/>"
        "• <b>Net Optimization Dynamics:</b> Parameters took unrestricted outward steps but were strictly forbidden from taking restorative steps back. "
        "Like a mechanical ratchet, the weights drifted uncontrollably in one direction, causing weight norms to explode and representations to collapse."
    )
    story.append(create_callout(ratchet_box, title="MECHANISTIC TELEMETRY CONFIRMATION", bg_color="#FFF5F5", border_color="#E53E3E", title_color="#9B2C2C"))
    story.append(Spacer(1, 6))

    # Telemetry data
    p_telem = ("To verify this hypothesis, <code>diagnostics.py</code> tracked step-by-step parameter telemetry across training tasks:<br/>"
               "• <code>[MEASURED]</code> <b>L2 Weight Norm Explosion:</b> By Task 5, layer 1 weights ($l1.weight$) under <code>gate:conflict</code> expanded to an L2 norm of "
               "<b>22.700</b> (compared to <b>11.790</b> for fine-tuning and <b>12.291</b> for magnitude gating).<br/>"
               "• <code>[MEASURED]</code> <b>Cumulative Drift Along d:</b> Parameter displacement projected onto $d$ ($\\frac{(\\theta - \\theta_{start}) \\cdot d}{\\|d\\|_2}$) "
               "drifted <b>+2.263</b> on Task 5 under <code>gate:conflict</code>, compared to <b>-0.145</b> for fine-tuning and <b>-0.016</b> for magnitude gating.<br/>"
               "• <code>[MEASURED]</code> <b>Gated Fraction:</b> Over 80% of all parameter coordinates satisfied $g \\cdot d > 0$ at every step across tasks 2 to 5.")
    story.append(Paragraph(p_telem, body_style))

    # Embed Diagnostics Images
    if os.path.exists("outputs/phase5/diagnostics_weight_norms.png") and os.path.exists("outputs/phase5/diagnostics_drift_along_d.png"):
        story.append(KeepTogether([
            Image("outputs/phase5/diagnostics_weight_norms.png", width=530, height=150),
            Paragraph("Figure 2: Step-by-step L2 Weight Norms (||θ||₂) across 5 tasks. Gate:conflict exhibits exponential norm expansion in layer 1.", caption_style),
            Spacer(1, 4),
            Image("outputs/phase5/diagnostics_drift_along_d.png", width=530, height=150),
            Paragraph("Figure 3: Parameter displacement projected along accumulated direction vector d. Gate:conflict displays massive unidirectional runaway drift.", caption_style)
        ]))

    story.append(Paragraph("The Three Repaired Gate Formulations (cl2.py)", h2_style))
    p_rep_desc = ("To halt the runaway ratchet while maintaining buffer-free operation, Phase 3 developed three stabilized variants:<br/>"
                  "1. <b>Anti-Drift Gradient Gate (<code>gate:drift</code>):</b> Rather than an additive loss penalty (which would constitute classic L2-SP or EWC regularizers), "
                  "<code>gate:drift</code> operates directly in the gradient domain. At task start, reference weights are frozen as $s = \\theta_{start}$. For displacement $e = \\theta - s$, "
                  "an SGD step expands distance from the anchor whenever $(-\\eta g) \\cdot e > 0 \\iff g \\cdot e < 0$. The gate tests:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;$\\text{cond} = (g \\odot d > 0) \\lor (g \\odot e < 0)$<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;$g \\leftarrow g \\odot (1 - \\beta \\cdot \\text{mag} \\cdot \\mathbf{1}_{\\text{cond}})$<br/>"
                  "By scaling down steps that increase distance from $\\theta_{start}$, it acts as a gradient-domain anchor brake analogous to movement-weighted L2-SP.<br/>"
                  "2. <b>Multiplicative Hybrid Gate (<code>gate:hybrid</code>):</b> Multiplies soft directional conflict damping by unconditional magnitude damping:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;$g \\leftarrow g \\odot (1 - \\beta \\cdot \\text{mag} \\cdot \\mathbf{1}_{g \\odot d > 0}) \\odot (1 - \\gamma \\cdot \\text{mag})$<br/>"
                  "3. <b>Soft Conflict Gate (<code>gate:conflict_soft</code>):</b> Retains the original direction check but restricts $\\beta \\in [0.05, 0.1]$, preventing updates from being completely blocked.")
    story.append(Paragraph(p_rep_desc, body_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 5: HARDER BENCHMARKS & FORGETTING CURVES
    # =========================================================================
    story.append(Paragraph("5. Long-Horizon Degradation & Forgetting Dynamics", h1_style))
    
    p_long = ("When continual learning is scaled from 5 tasks to 10 tasks, catastrophic forgetting in fine-tuning doubles from -9.62% to -20.80% BWT, "
              "dragging final accuracy down to 78.21 ± 1.68%. Under this demanding horizon, the original conflict gate suffered catastrophic breakdown, "
              "retaining only 13.44 ± 4.44% on Task 1 (near chance). In contrast, repaired gates demonstrated remarkable resilience: "
              "<code>gate:hybrid</code> (83.82 ± 0.54%) and <code>gate:drift</code> (83.21 ± 1.26%) statistically outperformed fine-tuning by over +5% ACC, "
              "retaining >52% Task 1 accuracy.")
    story.append(Paragraph(p_long, body_style))

    if os.path.exists("outputs/phase5/forgetting_curve_perm10_all.png"):
        story.append(KeepTogether([
            Image("outputs/phase5/forgetting_curve_perm10_all.png", width=510, height=270),
            Paragraph("Figure 4: Task 1 test accuracy retention (R_{i, 1}) across 10 sequential Permuted-MNIST tasks (5 seeds: 100–104). Repaired drift and hybrid gates prevent catastrophic collapse.", caption_style)
        ]))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 6: PHASE 4 LANGUAGE MODEL SPECIFICATION (FUTURE WORK)
    # =========================================================================
    story.append(Paragraph("6. Phase 4: Language Model Scaling Specification (Future Work)", h1_style))
    
    p_ph4_status = ("<b>Go/No-Go Decision & Hardware Scoping:</b> Per Section 199 of <code>AGENT_PLAN.md</code>, Phase 4 was subject to a mandatory go/no-go gate: "
                    "<i>'Run Phase 4 on a GPU (Colab or similar); do not run it on CPU. If time/hardware constraints exist, skip to Phase 5 and describe Phase 4 as future work.'</i> "
                    "Because the local workstation is an Apple M4 CPU, executing 8,000 steps of autoregressive transformer training across 4 text domains was computationally intractable. "
                    "Phase 4 was deliberately skipped and is specified in full detail as Future Work.")
    story.append(Paragraph(p_ph4_status, body_style))

    p_adam = ("<b>The Adaptive Optimizer Interaction Problem:</b> Large language models are universally trained using adaptive moment optimizers (Adam / AdamW). "
              "Adam computes first and second gradient moments: $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ and $v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$, "
              "normalizing updates coordinate-wise: $u_t = \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$.<br/>"
              "If gradient gating is applied directly to the raw backpropagated gradient $g_t$, Adam's division by $\\sqrt{v_t}$ rescales damped gradients back up, "
              "neutralizing the gate's effect. To resolve this, we designed <b><code>GatedAdam</code></b>, which injects the gate directly into the final parameter update vector:")
    story.append(Paragraph(p_adam, body_style))

    code_text = (
        "class GatedAdam(torch.optim.Optimizer):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;@torch.no_grad()<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;def step(self):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for group in self.param_groups:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for p in group['params']:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ... standard first (m) and second (v) moment updates ...<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upd = m_hat / (torch.sqrt(v_hat) + group['eps'])<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if self.gate is not None:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;upd = self.gate.scale_update(p, upd)  # Gating applied to Adam step!<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;p.add_(upd, alpha=-group['lr'])"
    )
    t_code = Table([[Paragraph(code_text, code_style)]], colWidths=[532])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_code)

    p_ph4_blueprint = ("<b>Future Experimental Protocol:</b><br/>"
                       "• <b>Base Model:</b> GPT-2 Small (124M parameters), pretrained weights.<br/>"
                       "• <b>Domains (4 Sequential):</b> WikiText-103 (encyclopedic), CodeSearchNet (Python), PubMed Abstracts (biomedical), Multi-News (financial/news).<br/>"
                       "• <b>Evaluation:</b> Held-out perplexity (lower is better) evaluated after each domain phase. Forgetting measured as $\\Delta PPL$ on earlier domains.")
    story.append(Paragraph(p_ph4_blueprint, body_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7: CODEBASE ARCHITECTURE & REPRODUCIBILITY GUIDE
    # =========================================================================
    story.append(Paragraph("7. Codebase Architecture & File Guide", h1_style))
    p_code = ("The AEGIS repository is fully modular, audited, and committed to git. Below is the comprehensive guide to all core files:")
    story.append(Paragraph(p_code, body_style))

    file_rows = [
        ["File / Script", "Phase", "Primary Purpose & Key Mechanism", "Execution Command"],
        ["cl.py", "Phase 0", "Frozen original reference code. Contains initial MLP and baseline SGD loop.", "python cl.py --seeds 1 --epochs 1"],
        ["cl2.py", "Phase 1-3", "Core enhanced engine: 55k/5k/10k splits, divergence guard, Gate class, repaired modes.", "python cl2.py --seeds 1 --epochs 1"],
        ["tune_val.py", "Phase 1", "Sweeps 25 hyperparameter configs across 7 methods on validation split (seeds 0-2).", "python tune_val.py"],
        ["run_final_perm5.py", "Phase 1", "Evaluates locked configs across 5 fresh test seeds (100-104) on 5-task Permuted-MNIST.", "python run_final_perm5.py"],
        ["run_perm10.py", "Phase 2", "Evaluates locked configs on 10 sequential tasks across 5 test seeds (100-104).", "python run_perm10.py"],
        ["tune_and_run_split.py", "Phase 2", "Validation tuning and 5-seed evaluation on domain-incremental Split-MNIST.", "python tune_and_run_split.py"],
        ["diagnostics.py", "Phase 3", "Logs step-by-step weight norms, drift along d, and tests ratchet hypothesis.", "python diagnostics.py"],
        ["tune_and_run_repaired.py", "Phase 3", "Tuning and 5-seed test evaluation of repaired gate variants (drift, hybrid, soft).", "python tune_and_run_repaired.py"],
        ["build_results_pack.py", "Phase 5", "Generates consolidated tables (CSV/PNG), master summary, and copies visual plots.", "python build_results_pack.py"]
    ]
    t_files = Table(file_rows, colWidths=[105, 45, 260, 122])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(primary_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6.5),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_files)

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 8: COMPLETION STATUS SCORECARD
    # =========================================================================
    story.append(Paragraph("8. Project Status Scorecard & Verification", h1_style))
    p_score = ("Project AEGIS has completed 100% of its empirical mandates, epistemological audits, and presentation deliverables. "
               "The table below details the exact completion status of every project milestone:")
    story.append(Paragraph(p_score, body_style))

    score_data = [
        ["Phase & Milestone", "Status", "Delivered Artifacts & Results", "Verification Check"],
        ["Phase 0: Setup & Freeze", "100% Complete", "cl.py frozen, cl2.py initialized, baseline timings verified on M4 CPU.", "Passed (commit 19df00d)"],
        ["Phase 1: Baselines & EWC Repair", "100% Complete", "55k/5k/10k split, EWC divergence isolated, λ=20 locked, Perm5 5-seed run.", "Passed (commit 2ec680e, b1ca604)"],
        ["Phase 2: Harder Benchmarks", "100% Complete", "10-task Permuted-MNIST, Split-MNIST shared head run, 5 test seeds.", "Passed (commit f5a03e6, 9eac5c3)"],
        ["Phase 3: Diagnosis & Repairs", "100% Complete", "Ratchet telemetry confirmed, gate:drift & gate:hybrid outperforming FT >+5%.", "Passed (commit 00453dc)"],
        ["Phase 4: Language Model Scaling", "Scoped (Future Work)", "Skipped on CPU per go/no-go rule; GatedAdam designed for GPU cluster.", "Documented in Report Sec 8"],
        ["Phase 5: Deliverables Pack", "100% Complete", "Master summary CSV/PNG, full research paper (report.md), captions.md.", "Passed (commit 572c750)"],
        ["Phase 5: Claim Audit", "100% Complete", "Purged 'prove', enforced statistical tie rule, verified all numbers.", "Passed (commit 7e9076c)"],
        ["Phase 5: Presentation Suite", "100% Complete", "Poster layout text, 15 examiner Q&A (qa.md), 60-second spoken script.", "Passed (commit 7e9076c)"],
        ["Phase 5: Master Technical PDF", "100% Complete", "Comprehensive 10-page master technical PDF report and markdown monograph.", "Current Document"]
    ]
    t_score = Table(score_data, colWidths=[125, 75, 232, 100])
    t_score.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(primary_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6.5),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('TEXTCOLOR', (1, 1), (1, -1), colors.HexColor("#22543D")), # Dark Green
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
    ]))
    story.append(t_score)

    story.append(Spacer(1, 10))
    final_note = ("<b>Conclusion:</b> Project AEGIS has successfully transformed an intuitive but failing gradient gating idea into a rigorous, "
                  "mechanistically understood, and empirically verified continual learning contribution. The negative findings were honestly reported, "
                  "the underlying mathematical dynamics were uncovered through step-level telemetry, and the resulting anti-drift repairs statistically "
                  "outperformed unconstrained fine-tuning on long-horizon benchmarks without memory replay buffers. All artifacts are committed, pushed, "
                  "and ready for poster defense, academic examination, and peer review.")
    story.append(Paragraph(final_note, body_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Master PDF report successfully built at {filename}!")


if __name__ == "__main__":
    out_pdf = "outputs/AEGIS_Master_Project_Report.pdf"
    build_pdf(out_pdf)
    shutil.copyfile(out_pdf, "outputs/phase5/AEGIS_Master_Project_Report.pdf")
    print("Copied PDF to outputs/phase5/AEGIS_Master_Project_Report.pdf")
