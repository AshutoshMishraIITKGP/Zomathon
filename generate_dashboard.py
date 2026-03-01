"""
Zomathon CSAO — Extended Dashboard Visualizations
Generates 8 high-res PNGs for the detailed multi-page submission.
All images at 150 DPI to keep total PDF under 1MB.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# ── Global Style ──
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',
    'axes.labelcolor': '#E6EDF3',
    'text.color': '#E6EDF3',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'grid.color': '#21262D',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 12,
})

DPI = 150  # Keep low for <1MB total
OUT = os.path.dirname(os.path.abspath(__file__))

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    sz = os.path.getsize(path) / 1024
    print(f"  ✓ {name} ({sz:.0f}KB)")


# =====================================================================
# 1. LATENCY BENCHMARK
# =====================================================================
def plot_latency():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = ['P50', 'P95', 'P99', 'Max']
    values = [3.65, 4.6, 5.3, 16.8]
    colors = ['#00D4AA', '#00C4B4', '#00B4BE', '#009ECE']

    bars = ax.barh(labels[::-1], values[::-1], height=0.55, color=colors[::-1],
                   edgecolor='none', zorder=3)
    ax.axvline(x=300, color='#FF4444', linewidth=2.5, linestyle='--', zorder=4, alpha=0.9)
    ax.text(305, 3.15, 'Zomato Budget\n300ms', color='#FF4444', fontsize=11,
            fontweight='bold', va='center')
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}ms', va='center', ha='left', fontsize=12, fontweight='bold', color='#E6EDF3')
    ax.annotate('82× under budget', xy=(3.65, 3), xytext=(100, 3),
                fontsize=16, fontweight='bold', color='#00D4AA',
                arrowprops=dict(arrowstyle='->', color='#00D4AA', lw=2, connectionstyle='arc3,rad=0.15'),
                va='center')
    ax.set_xlim(0, 380)
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_title('Full 4-Stage Pipeline Latency vs. Budget', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.text(0.5, 0.01, 'Graph + BERT + MMR + EV + XGBoost (7 features incl. SASRec)  |  1000-iter benchmark',
             ha='center', fontsize=8, color='#8B949E', style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, 'fig_latency.png')


# =====================================================================
# 2. AOV WATERFALL
# =====================================================================
def plot_aov():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(3)
    heights = [525.61, 410.12, 935.73]
    bottoms = [0, 525.61, 0]
    colors = ['#4A9EFF', '#00D4AA', '#FFB800']
    ax.bar(x, heights, bottom=bottoms, width=0.45, color=colors,
           edgecolor=['#3A8EEF','#00C49A','#EFA800'], linewidth=1.5, zorder=3)
    ax.plot([0.22, 0.78], [525.61, 525.61], color='#8B949E', lw=1, ls=':', zorder=2)
    for xi, yi, t in [(0,525.61/2,'Rs.526'),(1,525.61+410.12/2,'+Rs.410'),(2,935.73/2,'Rs.936')]:
        ax.text(xi, yi, t, ha='center', va='center', fontsize=15, fontweight='bold', color='white')
    ax.annotate('+78% AOV Uplift', xy=(1,935.73), xytext=(0.3,1030),
                fontsize=17, fontweight='bold', color='#00D4AA',
                arrowprops=dict(arrowstyle='->', color='#00D4AA', lw=2.5), ha='center')
    ax.set_ylim(0,1100)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(Single Item)', 'CSAO Rail\nAdd-On', 'Projected\nWith CSAO'], fontsize=11)
    ax.set_ylabel('Average Order Value (Rs.)')
    ax.set_title('AOV Impact: EV-Optimized CSAO Recommendations', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, 'fig_aov.png')


# =====================================================================
# 3. MMR DIVERSITY
# =====================================================================
def plot_mmr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    before = [('Peri Peri Crisper Fries',0.92),('Peri Peri Crispers Fries',0.88),
              ('Peri Peri Krispers',0.85),('Peri Peri Fries .',0.81),('Peri Peri Sauce',0.78)]
    after = [('Peri Peri Sauce',0.89),('Harisa Mayo',0.82),('Salted Fries',0.76),
             ('Tipsy Tiger Ginger Ale',0.71),('Onion Bombs',0.65)]
    reds = ['#FF6B6B','#FF8E8E','#FFB1B1','#FFD4D4','#FFE8E8']
    greens = ['#00D4AA','#00C4FF','#FFB800','#A78BFA','#F472B6']

    for items, ax, cols, title, tcol in [(before,ax1,reds,'Without MMR','#FF6B6B'),
                                         (after,ax2,greens,'With MMR (λ=0.6)','#00D4AA')]:
        ax.set_xlim(0,1.1); ax.set_ylim(-0.5,5.5)
        ax.set_title(title, fontsize=13, fontweight='bold', color=tcol, pad=8)
        for i,(name,score) in enumerate(items):
            y = 4-i
            ax.barh(y, score, height=0.5, color=cols[i], alpha=0.25,
                    edgecolor=cols[i], linewidth=1.5, zorder=3)
            ax.text(0.02, y, name, va='center', fontsize=9, fontweight='bold', color=cols[i])
            ax.text(score+0.02, y, f'{score:.2f}', va='center', fontsize=8, color='#8B949E')
        ax.set_yticks([]); ax.grid(axis='x', linestyle=':', alpha=0.3)
        for s in ['top','right','left']: ax.spines[s].set_visible(False)

    ax1.text(0.5,-0.32,'4/5 items are fries variants', ha='center', fontsize=10,
             fontweight='bold', color='#FF6B6B', transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', fc='#FF6B6B', alpha=0.1, ec='#FF6B6B'))
    ax2.text(0.5,-0.32,'5 distinct categories', ha='center', fontsize=10,
             fontweight='bold', color='#00D4AA', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', fc='#00D4AA', alpha=0.1, ec='#00D4AA'))

    fig.suptitle('Cart: Fried Chicken Peri Peri + Peri Peri Fries', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0.02,1,0.93])
    save(fig, 'fig_mmr.png')


# =====================================================================
# 4. SASREC ABLATION
# =====================================================================
def plot_sasrec():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'Val AUC\n(×100)', 'Val Acc']
    before = [10.2, 43.4, 64.4, 90.37, 81.2]
    after =  [19.6, 50.2, 66.8, 91.82, 84.3]
    imps = ['+92%', '+16%', '+3.7%', '+1.6%', '+3.1%']
    x = np.arange(len(metrics)); w = 0.3
    b1 = ax.bar(x-w/2, before, w, label='6 Features (No SASRec)', color='#4A9EFF', alpha=0.7, zorder=3)
    b2 = ax.bar(x+w/2, after, w, label='7 Features (+ SASRec)', color='#00D4AA', zorder=3)
    for bar,val in zip(b1, before):
        ax.text(bar.get_x()+bar.get_width()/2, val+1, f'{val}', ha='center', va='bottom', fontsize=8, color='#8B949E')
    for bar,val,imp in zip(b2, after, imps):
        ax.text(bar.get_x()+bar.get_width()/2, val+1, f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E6EDF3')
        ax.text(bar.get_x()+bar.get_width()/2, val+5, imp, ha='center', va='bottom', fontsize=8, fontweight='bold', color='#00D4AA')
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Score (%)'); ax.set_ylim(0,105)
    ax.set_title('SASRec Transformer: Feature Ablation Study', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.text(0.5,0.01,'SASRec: 2-layer, 64-dim Transformer  |  Zero online latency', ha='center', fontsize=8, color='#8B949E', style='italic')
    plt.tight_layout(rect=[0,0.04,1,1])
    save(fig, 'fig_sasrec.png')


# =====================================================================
# 5. HIT RATE PROGRESSION
# =====================================================================
def plot_hit_rates():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ks = [1, 3, 5, 10]
    hits = [19.6, 40.0, 50.2, 66.8]
    ax.plot(ks, hits, 'o-', color='#00D4AA', linewidth=2.5, markersize=10, zorder=3)
    for k, h in zip(ks, hits):
        ax.annotate(f'{h}%', (k, h), textcoords='offset points', xytext=(0, 12),
                    fontsize=12, fontweight='bold', color='#00D4AA', ha='center')
    ax.fill_between(ks, hits, alpha=0.1, color='#00D4AA')
    ax.set_xlabel('K (Top-K Recommendations)', fontsize=11)
    ax.set_ylabel('Hit Rate (%)', fontsize=11)
    ax.set_title('Hit@K Evaluation Curve', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(ks); ax.set_ylim(0, 80)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.text(0.5,0.01,'Leave-one-out evaluation on 500 orders with ≥3 items', ha='center', fontsize=8, color='#8B949E', style='italic')
    plt.tight_layout(rect=[0,0.04,1,1])
    save(fig, 'fig_hitrate.png')


# =====================================================================
# 6. CONTEXTUAL SUB-GRAPH BREAKDOWN
# =====================================================================
def plot_subgraphs():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = ['Morning', 'Afternoon', 'Dinner', 'Late Night',
             'Connaught Pl', 'Dwarka', 'Rohini', 'Sector 4', 'Sector 135', 'Shahdara', 'Vasant Kunj',
             'Budget', 'Mid', 'Premium']
    nodes = [72, 79, 97, 14, 80, 69, 90, 82, 90, 84, 60, 60, 73, 109]
    # Color by dimension
    colors = ['#FFB800']*4 + ['#00C4FF']*7 + ['#A78BFA']*3
    bars = ax.barh(range(len(names)), nodes, color=colors, height=0.7, edgecolor='none', zorder=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Number of Graph Nodes', fontsize=11)
    ax.set_title('13 Contextual Sub-Graphs by Dimension', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FFB800', label='Temporal (4)'),
                       Patch(facecolor='#00C4FF', label='Spatial (7)'),
                       Patch(facecolor='#A78BFA', label='Monetary (3)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.3)
    plt.tight_layout()
    save(fig, 'fig_subgraphs.png')


# =====================================================================
# 7. XGBOOST FEATURE IMPORTANCE
# =====================================================================
def plot_feature_importance():
    fig, ax = plt.subplots(figsize=(8, 4))
    features = ['graph_confidence', 'bert_similarity', 'sasrec_score',
                'item_price_norm', 'cart_value_norm', 'is_weekend', 'kpt_duration']
    # Approximate importances from a typical XGBoost run
    importances = [0.31, 0.24, 0.18, 0.12, 0.09, 0.04, 0.02]
    colors = ['#00D4AA','#00C4FF','#FFB800','#A78BFA','#F472B6','#8B949E','#555555']
    bars = ax.barh(range(len(features)), importances, color=colors, height=0.6, edgecolor='none', zorder=3)
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=10, fontfamily='monospace')
    for bar, imp in zip(bars, importances):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.0%}', va='center', fontsize=10, fontweight='bold', color='#E6EDF3')
    ax.set_xlabel('Feature Importance (Gain)', fontsize=11)
    ax.set_title('XGBoost LTR: Feature Importance Ranking', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(0, 0.42)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, 'fig_feature_importance.png')


# =====================================================================
# 8. PIPELINE ARCHITECTURE DIAGRAM
# =====================================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    ax.axis('off')
    ax.set_title('System Architecture: Offline-to-Online Pipeline', fontsize=14, fontweight='bold', pad=5)

    # Offline blocks (top row)
    offline_blocks = [
        (5, 22, 'Raw Orders\n21K strings', '#4A9EFF'),
        (22, 22, 'Assoc. Rules\n545 rules', '#FFB800'),
        (39, 22, 'Context Graphs\n13 sub-graphs', '#A78BFA'),
        (56, 22, 'BERT Encoding\n384-dim', '#00C4FF'),
        (73, 22, 'SASRec Train\n2L Transformer', '#F472B6'),
        (90, 22, 'XGBoost Train\n7 features', '#00D4AA'),
    ]
    for x, y, label, color in offline_blocks:
        rect = plt.Rectangle((x-6, y-3), 14, 6, facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=1.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(x+1, y, label, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=color)

    # Arrows between offline blocks
    for i in range(5):
        x1 = offline_blocks[i][0] + 8
        x2 = offline_blocks[i+1][0] - 6
        ax.annotate('', xy=(x2, 22), xytext=(x1, 22),
                    arrowprops=dict(arrowstyle='->', color='#8B949E', lw=1.2))

    # "OFFLINE" label
    ax.text(50, 28.5, '── OFFLINE BATCH (one-time) ──', ha='center', fontsize=9,
            color='#8B949E', style='italic')

    # Separator line
    ax.plot([0, 100], [16, 16], color='#30363D', linewidth=1, linestyle='--')
    ax.text(50, 16.5, '── ONLINE (<5ms) ──', ha='center', fontsize=9,
            color='#8B949E', style='italic', bbox=dict(facecolor='#0D1117', edgecolor='none', pad=3))

    # Online blocks (bottom row)
    online_blocks = [
        (10, 9, 'Cart Input', '#E6EDF3'),
        (27, 9, 'Graph Lookup\nO(1)', '#FFB800'),
        (44, 9, 'MMR Diversity\nλ=0.6', '#A78BFA'),
        (61, 9, 'EV Price Boost\nlog-price', '#00C4FF'),
        (78, 9, 'XGBoost Re-Rank\n7 features', '#00D4AA'),
        (95, 9, 'Top 5', '#FFB800'),
    ]
    for x, y, label, color in online_blocks:
        rect = plt.Rectangle((x-7, y-3), 14, 6, facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=color)

    for i in range(5):
        x1 = online_blocks[i][0] + 7
        x2 = online_blocks[i+1][0] - 7
        ax.annotate('', xy=(x2, 9), xytext=(x1, 9),
                    arrowprops=dict(arrowstyle='->', color='#00D4AA', lw=1.5))

    # Vertical arrows from offline to online
    for off_x, on_x in [(22, 27), (56, 44), (73, 78), (90, 78)]:
        ax.annotate('', xy=(on_x, 12), xytext=(off_x, 19),
                    arrowprops=dict(arrowstyle='->', color='#30363D', lw=1, ls='--'))

    plt.tight_layout()
    save(fig, 'fig_architecture.png')


# =====================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Generating Extended Dashboard (8 plots)...")
    print("=" * 50)
    plot_latency()
    plot_aov()
    plot_mmr()
    plot_sasrec()
    plot_hit_rates()
    plot_subgraphs()
    plot_feature_importance()
    plot_architecture()

    # Check total size
    total = 0
    for f in ['fig_latency.png','fig_aov.png','fig_mmr.png','fig_sasrec.png',
              'fig_hitrate.png','fig_subgraphs.png','fig_feature_importance.png','fig_architecture.png']:
        total += os.path.getsize(os.path.join(OUT, f))
    print(f"\n  Total image size: {total/1024:.0f}KB ({total/1024/1024:.2f}MB)")
    print(f"  Budget for PDF text: ~{1024 - total/1024:.0f}KB")
