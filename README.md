# Zomathon — CSAO Recommendation Engine

**Context-Aware, Two-Stage Recommendation Engine for Cart Super Add-Ons (CSAO)**

> Zomathon 2026 | Team: Prajyot · Ashish · Ashutosh Mishra

## Problem

Predict the next best item a customer will add to their Zomato cart, optimizing for **Average Order Value (AOV)**, **Cart-to-Order ratio (C2O)**, and **CSAO rail attach rate** — all under a strict **300ms latency budget**.

## Solution: Two-Stage Pipeline

```
Cart Input → Graph Lookup (O(1)) → MMR Diversity → EV Price Boost → XGBoost Re-Rank → Top-5
```

### Stage 1: Candidate Generation
- **Asymmetric Directed Graphs** via Association Rule Mining (Confidence, Lift, Support)
- **13 contextual sub-graphs** (Temporal × Spatial × Monetary)
- **BERT cold-start** (all-MiniLM-L6-v2, 384-dim) for new items with zero order history

### Stage 2: Re-Ranking Ensemble
- **SASRec Transformer** (2-layer, 64-dim) — offline transition score matrix
- **XGBoost LTR** (100 trees, 7 features) — final P(added | cart_state)
- **MMR** (λ=0.6) — diversity guardrail
- **EV re-ranking** (w=0.3) — profitability guardrail

## Key Metrics

| Metric | Value |
|---|---|
| **P50 Latency** | 3.65ms (82× under budget) |
| **XGBoost Val AUC** | 0.9182 |
| **Hit@5** | 50.2% |
| **Projected AOV Uplift** | +78% (Rs.525 → Rs.935) |
| **CSAO Attach Rate** | 100% |

## Setup & Run

```bash
# Install dependencies
pip install pandas numpy xgboost scikit-learn sentence-transformers torch matplotlib seaborn

# Run the full pipeline
python csao_pipeline.py

# Run simulation examples
python run_simulations.py

# Generate dashboard plots
python generate_dashboard.py
```

## Dataset

[Kaggle: Food Delivery Order History Data](https://www.kaggle.com/datasets/sujalsuthar/food-delivery-order-history-data?select=order_history_kaggle_data.csv)

- 21,131 orders from Delhi NCR
- 244 unique menu items
- Features: order items, bill subtotal, delivery zone, timestamps

## Project Structure

```
├── csao_pipeline.py              # Full pipeline (7 components)
├── run_simulations.py            # 10 cart scenario demos
├── generate_dashboard.py         # Dashboard visualization generator
├── csao_submission.tex           # LaTeX submission document
├── csao_submission.pdf           # Compiled PDF
├── csao_evaluation_results.json  # Metrics output
├── csao_xgb_reranker.json        # Trained XGBoost model
├── csao_sasrec_transitions.json  # SASRec transition matrix
├── csao_semantic_embeddings.json # BERT embeddings
├── csao_global_graph.json        # Association rule graph
├── csao_sub_graphs.json          # 13 contextual sub-graphs
└── fig_*.png                     # Dashboard figures
```

## Architecture

```
Offline (Batch)                    Online (<5ms)
─────────────────                  ──────────────
Raw Orders (21K)                   Cart Input
  ↓                                  ↓
Association Rules → Graph JSON  →  O(1) Graph Lookup
  ↓                                  ↓
BERT Encoding → Embeddings JSON →  Cold-start Fallback
  ↓                                  ↓
SASRec Training → Scores JSON   →  MMR Diversity (λ=0.6)
  ↓                                  ↓
XGBoost Training → Model JSON  →  EV Price Boost (w=0.3)
                                     ↓
                                   XGBoost Re-Rank
                                     ↓
                                   Top-5 CSAO Items
```
