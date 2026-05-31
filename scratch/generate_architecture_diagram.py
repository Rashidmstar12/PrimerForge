import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define colors matching the dark premium scheme
DARK_BG  = "#0f172a"
PANEL_BG = "#1e293b"
ACCENT   = "#06b6d4"
ACCENT2  = "#4f46e5"
TEXT_CLR = "#e2e8f0"
GREEN    = "#4ade80"
ORANGE   = "#fb923c"
VIOLET   = "#a78bfa"

fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)

# Hide axes
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

def draw_box(x, y, w, h, title, items, color, title_color="white"):
    # Draw shadow
    shadow = patches.FancyBboxPatch(
        (x+0.05, y-0.05), w, h, 
        boxstyle="round,pad=0.15", 
        facecolor="#020617", edgecolor="none", alpha=0.5
    )
    ax.add_patch(shadow)
    
    # Draw main box
    box = patches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle="round,pad=0.15", 
        facecolor=PANEL_BG, edgecolor=color, linewidth=1.5
    )
    ax.add_patch(box)
    
    # Text placement
    ax.text(x + w/2, y + h - 0.1, title, color=title_color, fontsize=11, fontweight="bold", ha="center", va="top")
    
    # Draw sub-items
    for i, item in enumerate(items):
        item_y = y + h - 0.6 - i * 0.42
        # Small bullet indicator
        ax.plot(x + 0.25, item_y + 0.05, "o", color=color, markersize=4)
        # Bullet text
        ax.text(x + 0.45, item_y, item, color=TEXT_CLR, fontsize=9, ha="left", va="bottom")

def draw_arrow(x1, y1, x2, y2, label="", color=ACCENT):
    ax.annotate(
        "", 
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=15)
    )
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, color=color, fontsize=8.5, fontweight="bold", ha="center")

# 1. INPUT LAYER (Shifted down to y=4.2 to prevent overlap)
draw_box(0.5, 4.2, 2.2, 2.0, "1. Multi-Inputs", [
    "Pangenome FASTA",
    "Population VCF",
    "Taq Polymerase brand",
    "Salt Buffer [Na+], [Mg2+]"
], ORANGE)

# 2. BIOPHYSICS ENGINE (Shifted down to y=4.2)
draw_box(3.6, 4.2, 2.4, 2.0, "2. Biophysics Engine", [
    "Nearest-Neighbor dG",
    "Nussinov MFE Folding",
    "Secondary Structure Cap",
    "Taq 3' Mismatch Decay"
], ACCENT)

# 3. MACHINE LEARNING ENSEMBLE (Shifted slightly down to y=1.2)
draw_box(3.6, 1.2, 2.4, 2.0, "3. ML Ensemble Scorer", [
    "5-Booster Stacked GBDT",
    "PyTorch Deep MLP",
    "Epistemic Uncertainty",
    "Platt Sigmoid Calibration"
], VIOLET)

# 4. MATHEMATICAL OPTIMIZATION (Shifted down to y=4.2)
draw_box(7.0, 4.2, 2.4, 2.0, "4. Graph & DP Routers", [
    "Integer Linear Prog (ILP)",
    "Pairwise Dimer Matrix",
    "Dynamic Prog Tiling",
    "Uniformity CV_P Solver"
], GREEN)

# 5. LAB ADAPTATION (Shifted slightly down to y=1.2)
draw_box(7.0, 1.2, 2.4, 2.0, "5. EWC Adaptation", [
    "Lab-Adaptive Fine-Tune",
    "Fisher Information regularizer",
    "Anti-Forgetting Guard",
    "ECE Brier optimization"
], ACCENT2)

# 6. OUTPUT STAGE (Shifted down to y=2.7 to balance middle-alignment)
draw_box(10.1, 2.7, 1.6, 2.0, "6. Diagnostics", [
    "Assay Viability (AVI)",
    "Panel Synergy (PSII)",
    "Scheme Uniform (SCUI)",
    "Interactive Reports"
], ACCENT)

# DRAW FLOW ARROWS (All arrow y-coordinates updated to match box centers)
# Input to Biophysics & ML
draw_arrow(2.9, 5.2, 3.4, 5.2)
draw_arrow(2.9, 5.2, 3.4, 2.2)

# Biophysics & ML to Downstream Optimizer
draw_arrow(6.2, 5.2, 6.8, 5.2)
draw_arrow(6.2, 2.2, 6.8, 2.2)

# ML Scorer feedback to Biophysics / EWC
draw_arrow(4.8, 3.4, 4.8, 4.0, "Thermodynamic feedback", color=ACCENT)

# Optimizer & EWC to output
draw_arrow(9.6, 5.2, 10.0, 3.8)
draw_arrow(9.6, 2.2, 10.0, 3.6)

# Linking Optimizer & EWC
draw_arrow(8.2, 3.4, 8.2, 4.0, "Constraints", color=GREEN)

# Title & Subtitle (Shifted up to create perfect breathing room)
ax.text(6.0, 7.5, "PrimerForge: End-to-End System Architecture Blueprint", 
        color="white", fontsize=15, fontweight="bold", ha="center")
ax.text(6.0, 7.15, "Pangenome-Aware molecular engineering pipeline from input sequence to clinical diagnostics", 
        color="#94a3b8", fontsize=10, style="italic", ha="center")

plt.tight_layout()
plt.savefig("c:/Users/rashi/Desktop/PYTHON CODES/new 23/primer tool/publication_package/figure5.png", dpi=300, facecolor=DARK_BG)
plt.close()
print("Success generating figure5.png")
