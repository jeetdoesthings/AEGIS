import os
import sys
import shutil

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#2B6CB0"))
            self.drawString(36, 756, "PROJECT AEGIS")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4A5568"))
            self.drawString(108, 756, "— Executive Overview, Results & Completion Brief")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(36, 750, 612 - 36, 750)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 42, 612 - 36, 42)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(36, 30, "Project AEGIS | Continual Learning & Catastrophic Forgetting Benchmark | github.com/jeetdoesthings/AEGIS")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 30, page_str)
        self.restoreState()


def create_card(title, body, bg_color="#F7FAFC", border_color="#CBD5E0", title_color="#1A365D", width=540):
    styles = getSampleStyleSheet()
    t_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor(title_color),
        spaceAfter=3
    )
    b_style = ParagraphStyle(
        'CardBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748")
    )
    cell = [
        Paragraph(title, t_style),
        Paragraph(body, b_style)
    ]
    t = Table([[cell]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


def build_pdf(filename="outputs/final/AEGIS_Simplified_Project_Overview.pdf"):
    print(f"Building Simplified PDF overview at {filename}...")
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    primary = "#1A365D"   # Deep Navy
    secondary = "#2B6CB0" # Slate Blue
    body_col = "#2D3748"  # Charcoal
    accent_green = "#22543D"
    accent_red = "#9B2C2C"

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=23,
        textColor=colors.HexColor(primary),
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor(secondary),
        spaceAfter=8
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(primary),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor(secondary),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor(body_col),
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11,
        textColor=colors.HexColor(body_col),
        leftIndent=10,
        spaceAfter=2
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=2,
        spaceAfter=6
    )

    story = []

    # =========================================================================
    # PAGE 1: WHAT IT IS & WHAT WE HAVE DONE & COMPLETION STATUS
    # =========================================================================
    story.append(Paragraph("PROJECT AEGIS: EXECUTIVE OVERVIEW & STATUS BRIEF", title_style))
    story.append(Paragraph("Continual Learning, Catastrophic Forgetting, and Parameter-Space Gradient Gating", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(primary), spaceBefore=1, spaceAfter=8))

    # High-level Completion Box (Prominent at the top)
    completion_text = (
        "<b>Overall Project Completion: 100% of Core Benchmark & Deliverables Roadmap Complete.</b><br/>"
        "• <b>Phase 0 (Setup & Frozen Baseline):</b> 100% Complete (committed <code>19df00d</code>).<br/>"
        "• <b>Phase 1 (Baselines & EWC Fix):</b> 100% Complete (55k/5k/10k splits, EWC explosion solved, Perm5 test run, commit <code>2ec680e</code>).<br/>"
        "• <b>Phase 2 (Harder Benchmarks):</b> 100% Complete (10-task Permuted-MNIST and Split-MNIST runs, commit <code>f5a03e6</code>).<br/>"
        "• <b>Phase 3 (Diagnostics & Repairs):</b> 100% Complete (Ratchet mechanism diagnosed, repaired gates beating FT by >+5%, commit <code>00453dc</code>).<br/>"
        "• <b>Phase 4 (Language Model Scaling):</b> Formally scoped as <b>Future Work</b> per go/no-go rule on local CPU (GPT-2 blueprint designed).<br/>"
        "• <b>Phase 5 (Deliverables & Audits):</b> 100% Complete (Results pack, research report, claim audit, poster, Q&A script, Master PDF, commit <code>e3af034</code>)."
    )
    story.append(create_card("PROJECT STATUS AT A GLANCE: 100% COMPLETE ROADMAP", completion_text, bg_color="#E6FFFA", border_color="#319795", title_color="#234E52"))
    story.append(Spacer(1, 6))

    # 1. WHAT IS THIS PROJECT ABOUT?
    story.append(Paragraph("1. What is this project about?", h1_style))
    p1 = ("<b>The Core Problem (Catastrophic Forgetting):</b> When artificial neural networks are trained on task A, and then sequentially "
          "trained on task B, the gradient descent updates required to master task B overwrite the weights critical for task A. "
          "Within a few updates, the network completely forgets how to perform task A. This fundamental limitation is called catastrophic forgetting.<br/>"
          "<b>Why RAG and Context Windows Don't Fix This:</b> Retrieval-Augmented Generation (RAG) and long-context windows in Large Language Models (LLMs) "
          "only feed text into temporary prompt tokens. They leave the actual neural network weights static; they cannot learn new procedural skills, "
          "adapt vision encoders, or retain knowledge permanently once a session closes. True AI autonomy requires lifelong learning directly inside the weights.<br/>"
          "<b>The AEGIS Research Question:</b> Can we create a <i>buffer-free</i> mathematical gate on gradients that shields critical weights without "
          "storing old data (no memory replay buffers)? We investigated the <b>Gradient-Conflict Gate</b>: an algorithm that tracks historical parameter movement "
          "and attempts to damp updates that oppose past learning trajectories.")
    story.append(Paragraph(p1, body_style))

    # 2. WHAT HAVE WE DONE IN THIS PROJECT?
    story.append(Paragraph("2. What have we done in this project?", h1_style))
    p2 = ("The project was conducted in five sequential, rigorously documented stages:")
    story.append(Paragraph(p2, body_style))

    story.append(Paragraph("• <b>Stage 1: Diagnosed & Fixed Broken Baselines (Before cl2.py vs After):</b> The starting codebase (<code>cl.py</code>) had an EWC baseline "
                           "that mysteriously scored 9.80% accuracy. We uncovered the mathematical cause: default penalty λ=5000 with learning rate 0.01 caused "
                           "exponential gradient explosion by step 13 into NaNs. <code>torch.argmax</code> on NaNs defaulted to digit '0' (which happens to be exactly 9.8% of MNIST!). "
                           "We added divergence guards, swept λ, locked optimal λ=20 (achieving 93.76% ACC), froze <code>cl.py</code> permanently, and rebuilt the engine cleanly in <code>cl2.py</code>.", bullet_style))
    story.append(Paragraph("• <b>Stage 2: Established Honest, Leakage-Free Evaluation:</b> Partitioned MNIST into 55,000 train, 5,000 validation (held-out tuning), "
                           "and 10,000 test images. Tuned all 7 methods across 25 configurations with equal tuning budgets (≤6 configs per method) strictly on validation split.", bullet_style))
    story.append(Paragraph("• <b>Stage 3: Uncovered the 'Asymmetric Ratchet' Failure:</b> On 10 tasks, the original directional conflict gate (<code>gate:conflict</code>) collapsed "
                           "to 41.92% ACC (-36.29% below fine-tuning!). Using step-level telemetry (<code>diagnostics.py</code>), we discovered why: blocking opposing updates while letting "
                           "outward updates run unhindered acted as a mechanical ratchet—layer 1 weight norms doubled from 11.79 to 22.70 and ran away along d (+2.263 vs -0.145).", bullet_style))
    story.append(Paragraph("• <b>Stage 4: Invented Stabilized Anti-Drift Repairs:</b> Formulated an anti-drift gate (<code>gate:drift</code>, which brakes steps that expand distance "
                           "from starting weights) and a hybrid gate (<code>gate:hybrid</code>). These repairs eliminated parameter runaway: on 10 tasks, <code>gate:hybrid</code> (83.82%) "
                           "and <code>gate:drift</code> (83.21%) <b>statistically beat fine-tuning (78.21%) by >+5% ACC</b> without storing replay data.", bullet_style))
    story.append(Paragraph("• <b>Stage 5: Tested Multi-Benchmark Boundaries:</b> Discovered that on domain-incremental Split-MNIST, all buffer-free methods tie at ~62% ACC, "
                           "revealing that weight-space gating cannot resolve shared output-head interference without data replay (96.85%) or task routing (LoRA: 98.73%).", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: WHAT WERE THE RESULTS?
    # =========================================================================
    story.append(Paragraph("3. What were the results?", h1_style))
    p3 = ("Every method was evaluated across 5 independent random test seeds (100–104) on held-out test splits. "
          "Under our strict <b>statistical tie rule</b>, two methods tie if their mean ± std intervals overlap; 'beats' is only claimed for strictly disjoint distributions.")
    story.append(Paragraph(p3, body_style))

    # Results Table
    res_data = [
        ["Method", "Category", "Perm5 ACC (%)", "Perm10 ACC (%)", "Split ACC (%)", "Key Finding / Takeaway"],
        ["Fine-Tuning (Plain SGD)", "Baseline", "88.97 ± 1.85", "78.21 ± 1.68", "62.07 ± 1.00", "Reference baseline. Forgetting doubles on 10 tasks (-20.8% BWT)."],
        ["EWC (repaired, λ=20/100)", "Regularizer", "93.76 ± 0.36", "90.22 ± 0.87", "63.96 ± 1.39", "Outperforms all gates (+6.4% on Perm10); strong on permutations; ties on Split."],
        ["Replay (buffer=1000)", "Exemplar Buffer", "95.85 ± 0.11", "95.22 ± 0.09", "96.85 ± 0.30", "Top overall accuracy on all benchmarks; stores 1,000 raw samples."],
        ["LoRA (r=16/8, oracle ID)*", "Architecture", "94.30 ± 0.12", "94.05 ± 0.05", "98.73 ± 0.06", "Zero backward forgetting (0.0% BWT); requires test-time task ID."],
        ["Gate: Conflict (original, β=0.25)", "Original Gate", "81.73 ± 1.41", "41.92 ± 1.71", "62.28 ± 0.98", "Catastrophic failure (-36.3% on Perm10); ratchet runaway."],
        ["Gate: Magnitude (ablation, β=1.0)", "Ablation", "91.43 ± 0.86", "83.13 ± 1.22", "61.90 ± 0.36", "Beats conflict gate by +41% on Perm10; magnitude alone works."],
        ["Gate: Drift (anti-drift, β=0.1)", "Repaired Gate", "91.89 ± 1.03", "83.21 ± 1.26", "62.62 ± 1.03", "Statistically beats fine-tuning by ~5.0% on Perm10 without buffers."],
        ["Gate: Hybrid (conflict×mag, β=0.1)", "Repaired Gate", "90.98 ± 0.90", "83.82 ± 0.54", "62.78 ± 0.56", "Statistically beats fine-tuning by ~5.6% on Perm10 without buffers."]
    ]
    t_res = Table(res_data, colWidths=[120, 75, 70, 70, 65, 140])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(primary)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (4, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('TEXTCOLOR', (0, 6), (4, 7), colors.HexColor("#2B6CB0")),
        ('FONTNAME', (0, 6), (4, 7), 'Helvetica-Bold'),
    ]))
    story.append(t_res)
    story.append(Paragraph("*LoRA requires an oracle task ID at test time to route representations to task-specific adapters.", caption_style))
    story.append(Spacer(1, 4))

    # Embed Visual Chart
    if os.path.exists("outputs/phase5/forgetting_curve_perm10_all.png"):
        story.append(Image("outputs/phase5/forgetting_curve_perm10_all.png", width=500, height=215))
        story.append(Paragraph("Figure 1: Task 1 accuracy retention across 10 sequential tasks on Permuted-MNIST (5 seeds: 100–104). Notice that original gate:conflict collapses to 13.4%, while repaired gate:drift and gate:hybrid maintain >52% retention, outperforming fine-tuning.", caption_style))

    # Honest Comparative Framing Card
    honest_card = (
        "<b>Rigorous Empirical & Honest Comparative Framing:</b><br/>"
        "• <b>Repaired EWC Outperforms Every Gate:</b> Repaired EWC (Perm10 ACC 90.22 ± 0.87%) outperforms every gate variant (83.13–83.82%) by over 6.4 percentage points. The gate is not competitive with EWC on retention.<br/>"
        "• <b>Repaired Gates Beat Fine-Tuning by ~5 Points:</b> On 10 tasks, gate:hybrid (83.82%) and gate:drift (83.21%) statistically beat plain fine-tuning (78.21%) by about 5 points (+5.62% and +5.01%) without memory buffers.<br/>"
        "• <b>Split-MNIST Buffer-Free Ceiling (62–64%):</b> On domain-incremental Split-MNIST, no method without replay (96.85%) or oracle task ID (LoRA: 98.73%) beats 62–64% ACC. Fine-tuning (62.07%), EWC (63.96%), and all gates (61.90–62.78%) form a mutual statistical tie due to shared-head logit overwrite."
    )
    story.append(create_card("HONEST METHOD COMPARISON & PERFORMANCE BOUNDARIES", honest_card, bg_color="#FFF5F5", border_color="#E53E3E", title_color="#9B2C2C"))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: WHAT ARE THE OUTPUTS & COMPLETION DETAILS
    # =========================================================================
    story.append(Paragraph("4. What are the outputs?", h1_style))
    p4 = ("The project produced a complete, transparent, and reproducible suite of deliverables in <code>outputs/</code>:")
    story.append(Paragraph(p4, body_style))

    out_cards = (
        "<b>1. Published Master Documents:</b><br/>"
        "• <code>outputs/final/AEGIS_Master_Project_Report.pdf</code>: Comprehensive 6-page technical monograph with embedded figures.<br/>"
        "• <code>outputs/phase5/report.md</code>: Formal 8-section academic research paper draft.<br/>"
        "• <code>outputs/phase5/claim_audit.md</code>: Epistemological audit verifying all claims against raw numbers under the tie rule.<br/>"
        "• <code>outputs/phase5/poster_text.md</code>, <code>qa.md</code>, <code>script_60s.md</code>: Ready-to-present defense materials.<br/>"
        "<b>2. Benchmark Tables & Raw Results:</b><br/>"
        "• <code>outputs/phase5/master_summary.csv</code> and <code>master_summary.png</code>: 10-method cross-benchmark master comparison.<br/>"
        "• <code>outputs/phase5/table_perm5.*</code>, <code>table_perm10.*</code>, <code>table_split.*</code>: Per-benchmark publication tables.<br/>"
        "• <code>outputs/phase1/final_perm5.json</code>, <code>phase2/final_perm10.json</code>, <code>phase3/final_repaired_*.json</code>: Full raw test logs.<br/>"
        "<b>3. Visual Figures & Telemetry:</b><br/>"
        "• Forgetting Curves: <code>outputs/phase5/forgetting_curve_perm5_all.png</code>, <code>forgetting_curve_perm10_all.png</code>.<br/>"
        "• Accuracy Heatmaps: <code>outputs/phase5/heatmaps_perm5.png</code>, <code>heatmaps_perm10.png</code>, <code>heatmaps_split.png</code>.<br/>"
        "• Telemetry Plots: <code>diagnostics_weight_norms.png</code>, <code>diagnostics_drift_along_d.png</code>, <code>layer_heatmap_best_gate.png</code>."
    )
    story.append(create_card("COMPLETE DELIVERABLES CATALOG", out_cards, bg_color="#F7FAFC", border_color="#CBD5E0", title_color="#1A365D"))
    story.append(Spacer(1, 6))

    # 5. COMPLETION PERCENTAGE & FUTURE WORK
    story.append(Paragraph("5. What is the completion percentage?", h1_style))
    p5 = ("<b>Core Experimental Roadmap: 100% Complete.</b> Every empirical milestone across 5-task Permuted-MNIST, 10-task Permuted-MNIST, "
          "domain-incremental Split-MNIST, EWC diagnosis, parameter telemetry, and gate repairs has been executed across 5 independent seeds, audited, and committed.<br/>"
          "<b>Note on Phase 4 (Language Models):</b> Phase 4 (evaluating a 124M-parameter GPT-2 Small model across 4 text domains) was "
          "<b>scoped out to Future Work</b> per the project plan's go/no-go gate rule because execution was on local CPU hardware. Training an autoregressive "
          "transformer across 8,000 steps without GPU acceleration is computationally intractable. The theoretical foundation, the Adam moment interaction, "
          "and the <code>GatedAdam</code> optimizer are fully specified in Section 6 of the Master Report for immediate GPU deployment.")
    story.append(Paragraph(p5, body_style))

    # How to Reproduce Box
    repro_text = (
        "<b>Fast Reproduction Commands (Run in Terminal):</b><br/>"
        "1. <b>Quick Smoke Test (10s):</b> <code>python cl2.py --seeds 1 --epochs 1</code><br/>"
        "2. <b>Tune & Evaluate Repaired Gates:</b> <code>python tune_and_run_repaired.py</code><br/>"
        "3. <b>Run Telemetry Diagnostics:</b> <code>python diagnostics.py</code><br/>"
        "4. <b>Regenerate Master Tables & PDF:</b> <code>python build_results_pack.py && python build_master_pdf.py</code><br/>"
        "<b>Repository:</b> All code and data pushed to <font color='#2B6CB0'><u>https://github.com/jeetdoesthings/AEGIS</u></font>."
    )
    story.append(create_card("REPRODUCIBILITY & VERIFICATION COMMANDS", repro_text, bg_color="#EDF2F7", border_color="#A0AEC0", title_color="#2D3748"))
    story.append(Spacer(1, 6))

    final_quote = ("<b>Summary Takeaway:</b> Project AEGIS turned an initial prototype failure into a valuable scientific contribution: "
                   "by refusing to cherry-pick or hide negative results, we uncovered the mathematical failure mode of directional conflict gating (the ratchet effect), "
                   "demonstrated that magnitude gating outperforms direction checking, and engineered anti-drift repairs that statistically beat fine-tuning on long horizons without storing replay data.")
    story.append(Paragraph(final_quote, body_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Simplified PDF report successfully built at {filename}!")


if __name__ == "__main__":
    out_pdf = "outputs/final/AEGIS_Simplified_Project_Overview.pdf"
    build_pdf(out_pdf)
