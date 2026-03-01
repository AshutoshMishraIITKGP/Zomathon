"""
=============================================================================
ZOMATHON: Cart Super Add-On (CSAO) Rail Recommendation System
=============================================================================
Production-ready pipeline implementing Context-Aware Asymmetric Directed Graphs
using Association Rule Mining for real-time cart recommendations.

Core Metrics Optimized:
  - Average Order Value (AOV)
  - Cart-to-Order (C2O) Ratio
  - CSAO Rail Attach Rate

Latency Target: < 200-300ms (achieved via O(1) dictionary lookups)

Author: Zomathon Team
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import json
import time
import hashlib
from collections import defaultdict, Counter
from itertools import combinations, permutations
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COMPONENT 1: DATA CLEANING & ITEM PARSING
# ============================================================================

class DataProcessor:
    """Handles raw order data ingestion, item parsing, and feature engineering."""

    # Temporal daypart boundaries (hour-based)
    DAYPART_MAP = {
        'breakfast': (6, 11),     # 6 AM - 11 AM
        'lunch': (11, 15),        # 11 AM - 3 PM
        'snacks': (15, 18),       # 3 PM - 6 PM
        'dinner': (18, 23),       # 6 PM - 11 PM
        'late_night': (23, 6),    # 11 PM - 6 AM
    }

    # Bill subtotal buckets
    BUDGET_THRESHOLDS = {
        'budget': (0, 400),
        'mid': (400, 800),
        'premium': (800, float('inf')),
    }

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.item_catalog = set()
        self.parsed_orders = []
        self.item_avg_price = {}  # Estimated per-item price from order data

    def load_and_clean(self) -> pd.DataFrame:
        """Load CSV and perform initial cleaning."""
        print("[1/5] Loading raw dataset...")
        self.df = pd.read_csv(self.csv_path)

        print(f"  -> Raw rows: {len(self.df):,}")
        print(f"  -> Columns: {list(self.df.columns)}")

        # Filter to delivered orders only
        if 'Order Status' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['Order Status'] == 'Delivered'].copy()
            print(f"  -> Filtered to Delivered only: {len(self.df):,} (dropped {before - len(self.df):,})")

        # Parse timestamps
        self.df['Order Placed At'] = self.df['Order Placed At'].apply(self._parse_timestamp)

        # Clean bill subtotal
        self.df['Bill subtotal'] = pd.to_numeric(self.df['Bill subtotal'], errors='coerce').fillna(0)

        return self.df

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse Zomato's timestamp format: '11:38 PM, September 10 2024'"""
        try:
            return datetime.strptime(str(ts_str).strip(), "%I:%M %p, %B %d %Y")
        except (ValueError, TypeError):
            return None

    def parse_items(self) -> list:
        """
        Parse 'Items in order' column.
        Input:  "1 x Grilled Chicken Jamaican Tender, 2 x Coke"
        Output: ['Grilled Chicken Jamaican Tender', 'Coke'] (quantities stripped)
        """
        print("[2/5] Parsing items and building catalog...")

        for _, row in self.df.iterrows():
            items_str = str(row.get('Items in order', ''))
            if not items_str or items_str == 'nan':
                continue

            # Split by comma, then strip quantity prefix "N x "
            raw_items = [item.strip() for item in items_str.split(',')]
            parsed = []
            for item in raw_items:
                # Remove quantity prefix like "1 x ", "2 x "
                clean = re.sub(r'^\d+\s*x\s+', '', item, flags=re.IGNORECASE).strip()
                if clean:
                    parsed.append(clean)
                    self.item_catalog.add(clean)

            if len(parsed) >= 1:
                self.parsed_orders.append({
                    'order_id': row.get('Order ID', ''),
                    'restaurant_id': row.get('Restaurant ID', ''),
                    'restaurant_name': row.get('Restaurant name', ''),
                    'items': parsed,
                    'unique_items': list(set(parsed)),  # Deduplicated for graph edges
                    'num_items': len(parsed),
                    'timestamp': row.get('Order Placed At'),
                    'subzone': row.get('Subzone', 'Unknown'),
                    'city': row.get('City', 'Unknown'),
                    'bill_subtotal': row.get('Bill subtotal', 0),
                    'customer_id': row.get('Customer ID', ''),
                })

        print(f"  -> Unique items in catalog: {len(self.item_catalog):,}")
        print(f"  -> Parsed orders (≥1 item): {len(self.parsed_orders):,}")

        return self.parsed_orders

    def engineer_features(self) -> list:
        """Add temporal, spatial, and monetary context features to each order."""
        print("[3/5] Engineering contextual features...")

        for order in self.parsed_orders:
            ts = order['timestamp']

            # --- Temporal: Daypart ---
            if ts:
                hour = ts.hour
                order['daypart'] = self._get_daypart(hour)
                order['hour'] = hour
                order['day_of_week'] = ts.strftime('%A')
                order['is_weekend'] = ts.weekday() >= 5
            else:
                order['daypart'] = 'unknown'
                order['hour'] = -1
                order['day_of_week'] = 'unknown'
                order['is_weekend'] = False

            # --- Monetary: Budget Bucket ---
            bill = order['bill_subtotal']
            order['price_tier'] = self._get_price_tier(bill)

        daypart_dist = Counter(o['daypart'] for o in self.parsed_orders)
        tier_dist = Counter(o['price_tier'] for o in self.parsed_orders)
        print(f"  -> Daypart distribution: {dict(daypart_dist)}")
        print(f"  -> Price tier distribution: {dict(tier_dist)}")

        # --- Estimate per-item prices from bill / item_count ---
        self._estimate_item_prices()

        return self.parsed_orders

    def _estimate_item_prices(self):
        """
        Estimate per-item average price from bill_subtotal / num_items.
        This is an approximation since we don't have individual item prices.
        """
        item_price_sums = defaultdict(float)
        item_price_counts = defaultdict(int)

        for order in self.parsed_orders:
            if order['num_items'] > 0 and order['bill_subtotal'] > 0:
                avg_price_per_item = order['bill_subtotal'] / order['num_items']
                for item in order['unique_items']:
                    item_price_sums[item] += avg_price_per_item
                    item_price_counts[item] += 1

        self.item_avg_price = {
            item: round(item_price_sums[item] / item_price_counts[item], 2)
            for item in item_price_sums
            if item_price_counts[item] > 0
        }

        if self.item_avg_price:
            prices = list(self.item_avg_price.values())
            print(f"  -> Item price estimates: min=₹{min(prices):.0f}, "
                  f"median=₹{np.median(prices):.0f}, max=₹{max(prices):.0f}")

    def _get_daypart(self, hour: int) -> str:
        """Map hour to daypart."""
        if 6 <= hour < 11:
            return 'breakfast'
        elif 11 <= hour < 15:
            return 'lunch'
        elif 15 <= hour < 18:
            return 'snacks'
        elif 18 <= hour < 23:
            return 'dinner'
        else:
            return 'late_night'

    def _get_price_tier(self, bill: float) -> str:
        """Map bill subtotal to price tier."""
        if bill < 400:
            return 'budget'
        elif bill < 800:
            return 'mid'
        else:
            return 'premium'

    def get_multi_item_orders(self) -> list:
        """Return only orders with 2+ unique items (needed for co-occurrence)."""
        multi = [o for o in self.parsed_orders if len(o['unique_items']) >= 2]
        print(f"  -> Multi-item orders (≥2 unique items): {len(multi):,}")
        return multi


# ============================================================================
# COMPONENT 2: ASSOCIATION RULE MINING ENGINE
# ============================================================================

class AssociationRuleEngine:
    """
    Computes Support, Confidence, and Lift for all item pairs.
    Builds the Asymmetric Directed Graph where:
      - Nodes = unique food items
      - Edges = directional statistical relationships (A → B)
    """

    def __init__(self, orders: list, item_catalog: set, min_support: float = 0.001,
                 min_confidence: float = 0.05, min_lift: float = 1.0):
        self.orders = orders
        self.item_catalog = item_catalog
        self.total_orders = len(orders)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

        # Computed metrics
        self.item_support = {}       # P(A)
        self.pair_support = {}       # P(A ∩ B)
        self.confidence = {}         # P(B|A) = P(A ∩ B) / P(A)
        self.lift = {}               # Lift(A→B) = Confidence(A→B) / P(B)
        self.rules = []              # Final filtered rules

    def compute_support(self) -> dict:
        """Compute Support = P(item) for every item in catalog."""
        print("[4/5] Computing Association Rules...")
        print("  -> Step 1: Item Support P(A)...")

        item_counts = Counter()
        for order in self.orders:
            for item in order['unique_items']:
                item_counts[item] += 1

        self.item_support = {
            item: count / self.total_orders
            for item, count in item_counts.items()
        }

        # Print top-10 most popular items
        top_items = sorted(self.item_support.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  -> Top 10 items by Support:")
        for item, sup in top_items:
            print(f"     {item}: {sup:.4f} ({int(sup * self.total_orders)} orders)")

        return self.item_support

    def compute_pair_metrics(self) -> tuple:
        """Compute pairwise Support, Confidence, and Lift."""
        print("  -> Step 2: Pair Support P(A ∩ B)...")

        pair_counts = Counter()
        for order in self.orders:
            items = order['unique_items']
            if len(items) < 2:
                continue
            # All unique ordered pairs (directional)
            for a, b in permutations(items, 2):
                pair_counts[(a, b)] += 1

        # Compute pair support
        self.pair_support = {
            pair: count / self.total_orders
            for pair, count in pair_counts.items()
        }

        print(f"  -> Total directional pairs found: {len(self.pair_support):,}")

        # Compute Confidence and Lift
        print("  -> Step 3: Confidence P(B|A) and Lift...")

        for (a, b), p_ab in self.pair_support.items():
            p_a = self.item_support.get(a, 0)
            p_b = self.item_support.get(b, 0)

            if p_a == 0 or p_b == 0:
                continue

            conf = p_ab / p_a  # Confidence(A → B)
            lift_val = conf / p_b  # Lift(A → B)

            self.confidence[(a, b)] = conf
            self.lift[(a, b)] = lift_val

        return self.confidence, self.lift

    def filter_rules(self) -> list:
        """Apply minimum thresholds and build final rule set."""
        print("  -> Step 4: Filtering rules (Lift > 1.0)...")

        self.rules = []
        for (a, b), lift_val in self.lift.items():
            conf = self.confidence.get((a, b), 0)
            sup = self.pair_support.get((a, b), 0)

            if (sup >= self.min_support and
                conf >= self.min_confidence and
                lift_val > self.min_lift):  # Strictly > 1 to avoid trivial

                self.rules.append({
                    'antecedent': a,
                    'consequent': b,
                    'support': round(sup, 6),
                    'confidence': round(conf, 4),
                    'lift': round(lift_val, 4),
                    'pair_count': int(sup * self.total_orders),
                })

        # Sort by lift descending
        self.rules.sort(key=lambda x: x['lift'], reverse=True)

        print(f"  -> Rules passing all thresholds: {len(self.rules):,}")
        print(f"  -> Top 10 rules by Lift:")
        for r in self.rules[:10]:
            print(f"     {r['antecedent']} → {r['consequent']}  "
                  f"[Conf={r['confidence']:.2f}, Lift={r['lift']:.2f}, "
                  f"Count={r['pair_count']}]")

        return self.rules

    def build_graph(self) -> dict:
        """
        Build the adjacency-list representation of the Directed Graph.
        Format: { item_A: [ {item: B, confidence: X, lift: Y, support: Z}, ... ] }
        Sorted by composite score for each source node.
        """
        graph = defaultdict(list)

        for rule in self.rules:
            graph[rule['antecedent']].append({
                'item': rule['consequent'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'support': rule['support'],
                'score': round(rule['confidence'] * rule['lift'], 4),  # Composite score
            })

        # Sort each node's edges by composite score descending
        for node in graph:
            graph[node].sort(key=lambda x: x['score'], reverse=True)

        print(f"  -> Graph nodes with outgoing edges: {len(graph):,}")
        avg_edges = np.mean([len(v) for v in graph.values()]) if graph else 0
        print(f"  -> Avg outgoing edges per node: {avg_edges:.1f}")

        return dict(graph)


# ============================================================================
# COMPONENT 3: CONTEXTUAL SUB-GRAPH CONSTRUCTION
# ============================================================================

class ContextualGraphBuilder:
    """
    Builds context-specific sub-graphs sliced by:
      1. Temporal (daypart)
      2. Spatial (subzone)
      3. Monetary (price tier)
    """

    def __init__(self, orders: list, item_catalog: set,
                 min_support: float = 0.001,
                 min_confidence: float = 0.03,
                 min_lift: float = 1.0):
        self.orders = orders
        self.item_catalog = item_catalog
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.sub_graphs = {}

    def build_all_sub_graphs(self) -> dict:
        """Build sub-graphs for all context dimensions."""
        print("\n[5/5] Building Contextual Sub-Graphs...")

        # --- Temporal Sub-Graphs ---
        dayparts = set(o['daypart'] for o in self.orders if o['daypart'] != 'unknown')
        for dp in sorted(dayparts):
            subset = [o for o in self.orders if o['daypart'] == dp]
            if len(subset) >= 10:  # Minimum orders for meaningful graph
                key = f"temporal_{dp}"
                graph = self._build_subgraph(subset, key)
                if graph:
                    self.sub_graphs[key] = graph

        # --- Spatial Sub-Graphs ---
        subzones = set(o['subzone'] for o in self.orders if o['subzone'] != 'Unknown')
        for sz in sorted(subzones):
            subset = [o for o in self.orders if o['subzone'] == sz]
            if len(subset) >= 10:
                key = f"spatial_{sz.replace(' ', '_').lower()}"
                graph = self._build_subgraph(subset, key)
                if graph:
                    self.sub_graphs[key] = graph

        # --- Monetary Sub-Graphs ---
        tiers = set(o['price_tier'] for o in self.orders)
        for tier in sorted(tiers):
            subset = [o for o in self.orders if o['price_tier'] == tier]
            if len(subset) >= 10:
                key = f"monetary_{tier}"
                graph = self._build_subgraph(subset, key)
                if graph:
                    self.sub_graphs[key] = graph

        print(f"\n  => Total sub-graphs built: {len(self.sub_graphs)}")
        for k, v in self.sub_graphs.items():
            print(f"     {k}: {len(v)} nodes")

        return self.sub_graphs

    def _build_subgraph(self, orders: list, label: str) -> dict:
        """Build a single sub-graph from a filtered order subset."""
        multi_item = [o for o in orders if len(o['unique_items']) >= 2]
        if len(multi_item) < 5:
            return {}

        engine = AssociationRuleEngine(
            orders=orders,
            item_catalog=self.item_catalog,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_lift=self.min_lift,
        )
        # Suppress prints for sub-graphs
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        engine.compute_support()
        engine.compute_pair_metrics()
        engine.filter_rules()
        graph = engine.build_graph()

        sys.stdout = old_stdout
        return graph


# ============================================================================
# COMPONENT 3.5: BERT SEMANTIC ENCODER
# ============================================================================

class SemanticEncoder:
    """
    BERT-based semantic encoding for food items.
    Uses sentence-transformers (all-MiniLM-L6-v2) to embed item names into
    384-dimensional dense vectors, enabling:
      1. Cosine similarity between any two items
      2. Automatic semantic clustering
      3. Cold-start resolution via nearest-neighbor lookup
    """

    MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim, fast, multilingual-capable

    def __init__(self, item_catalog: set):
        from sentence_transformers import SentenceTransformer
        
        self.items = sorted(list(item_catalog))
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        
        print("\n=== BERT SEMANTIC ENCODER ===")
        print(f"  Loading model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        
        # Encode all items
        print(f"  Encoding {len(self.items)} menu items...")
        self.embeddings = self.model.encode(
            self.items,
            show_progress_bar=False,
            normalize_embeddings=True,  # Unit vectors for cosine sim via dot product
            batch_size=64,
        )
        
        # Pre-compute full similarity matrix (N×N) — O(N²) but N=244 so trivial
        self.similarity_matrix = self.embeddings @ self.embeddings.T
        
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Similarity matrix: {self.similarity_matrix.shape}")
        
        # Auto-discover clusters
        self.clusters = self._cluster_items()
        print(f"  Semantic clusters discovered: {len(self.clusters)}")
        for cluster_name, members in sorted(self.clusters.items()):
            print(f"    {cluster_name}: {len(members)} items")

    def get_embedding(self, item: str) -> np.ndarray:
        """Get pre-computed embedding for a catalog item."""
        if item in self.item_to_idx:
            return self.embeddings[self.item_to_idx[item]]
        # Encode on-the-fly for unknown items (cold start)
        return self.model.encode([item], normalize_embeddings=True)[0]

    def get_similarity(self, item_a: str, item_b: str) -> float:
        """Compute cosine similarity between two items."""
        if item_a in self.item_to_idx and item_b in self.item_to_idx:
            return float(self.similarity_matrix[
                self.item_to_idx[item_a],
                self.item_to_idx[item_b]
            ])
        # Fallback: compute directly from embeddings
        emb_a = self.get_embedding(item_a)
        emb_b = self.get_embedding(item_b)
        return float(np.dot(emb_a, emb_b))

    def get_nearest_neighbors(self, item: str, top_k: int = 5,
                                exclude_self: bool = True) -> list:
        """
        Find the top-K most semantically similar items.
        Returns list of (item_name, similarity_score) tuples.
        """
        emb = self.get_embedding(item)
        similarities = emb @ self.embeddings.T  # (N,) dot products = cosine sims
        
        # Get top indices
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            neighbor = self.items[idx]
            sim = float(similarities[idx])
            if exclude_self and neighbor == item:
                continue
            results.append((neighbor, sim))
            if len(results) >= top_k:
                break
        
        return results

    def encode_new_item(self, item_name: str) -> tuple:
        """
        Cold-start handler: encode a never-seen item and find its
        nearest catalog neighbor to inherit graph edges from.
        """
        neighbors = self.get_nearest_neighbors(item_name, top_k=3)
        return neighbors

    def _cluster_items(self, threshold: float = 0.65) -> dict:
        """
        Auto-discover semantic clusters using agglomerative approach.
        Items with similarity > threshold are grouped together.
        """
        from collections import defaultdict
        
        visited = set()
        clusters = defaultdict(list)
        
        for i, item in enumerate(self.items):
            if item in visited:
                continue
            
            # Find all items similar to this one
            cluster_members = [item]
            visited.add(item)
            
            for j, other in enumerate(self.items):
                if other in visited or i == j:
                    continue
                if self.similarity_matrix[i, j] > threshold:
                    cluster_members.append(other)
                    visited.add(other)
            
            if len(cluster_members) >= 2:
                # Name cluster by shortest common substring or first item
                cluster_name = self._infer_cluster_name(cluster_members)
                clusters[cluster_name] = cluster_members
        
        return dict(clusters)

    def _infer_cluster_name(self, members: list) -> str:
        """Infer a human-readable cluster name from member items."""
        # Find most common word across members
        word_counts = Counter()
        for item in members:
            words = item.lower().split()
            word_counts.update(words)
        
        # Filter out generic words
        stop_words = {'x', 'in', 'the', 'a', 'an', 'of', 'with', 'and', '1', '2', '+'}
        top_words = [
            word for word, _ in word_counts.most_common(3)
            if word not in stop_words and len(word) > 2
        ]
        
        return ' '.join(top_words[:2]).title() if top_words else members[0][:20]

    def export_embeddings(self, output_path: str):
        """Export embeddings and similarity data for visualization/caching."""
        export_data = {
            'model': self.MODEL_NAME,
            'embedding_dim': int(self.embeddings.shape[1]),
            'num_items': len(self.items),
            'clusters': self.clusters,
            'items': self.items,
            'sample_similarities': []
        }
        
        # Add top-3 neighbors for each item as a sample
        for item in self.items[:20]:  # Sample first 20
            neighbors = self.get_nearest_neighbors(item, top_k=3)
            export_data['sample_similarities'].append({
                'item': item,
                'neighbors': [{'item': n, 'similarity': round(s, 4)} for n, s in neighbors]
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"  -> Saved semantic data: {output_path}")


# ============================================================================
# COMPONENT 4: REAL-TIME INFERENCE ENGINE (O(1) LOOKUP)
# ============================================================================

class RecommendationEngine:
    """
    Real-time inference engine that serves CSAO recommendations.
    Performs O(1) dictionary lookups against pre-computed graphs.
    """

    def __init__(self, global_graph: dict, sub_graphs: dict, item_catalog: set,
                 semantic_encoder: SemanticEncoder = None,
                 item_avg_price: dict = None,
                 xgb_reranker = None,
                 mmr_lambda: float = 0.6,
                 ev_weight: float = 0.3):
        self.global_graph = global_graph
        self.sub_graphs = sub_graphs
        self.item_catalog = item_catalog
        self.semantic_encoder = semantic_encoder
        self.item_avg_price = item_avg_price or {}
        self.xgb_reranker = xgb_reranker
        self.mmr_lambda = mmr_lambda  # MMR diversity-relevance tradeoff
        self.ev_weight = ev_weight    # EV price boost weight

        # Pre-compute log-scaled normalized prices for EV re-ranking
        self.normalized_prices = self._normalize_prices()

        # Build category mapping as lightweight fallback (if BERT unavailable)
        self.category_map = self._build_category_map()

    def _build_category_map(self) -> dict:
        """
        NLP-based semantic fallback for cold start.
        Maps items to inferred parent categories using keyword extraction.
        """
        category_keywords = {
            'chicken': ['chicken', 'tender', 'tangdi', 'grilled', 'fried', 'bone', 'wings'],
            'pizza': ['pizza', 'melt', 'pide'],
            'fries': ['fries', 'krispers'],
            'sides': ['slaw', 'onion', 'potato', 'herbed', 'bombs'],
            'rice': ['rice', 'biryani'],
            'sauce': ['sauce', 'aioli', 'mayo', 'dip', 'harisa', 'cafreal', 'pico'],
            'beverage': ['ale', 'soda', 'tea', 'tiger', 'tipsy', 'coke', 'pepsi', 'drink'],
            'bread': ['garlic bread', 'bread'],
            'dessert': ['peanut', 'chocolate', 'malai', 'gud'],
            'paneer': ['paneer'],
        }

        cat_map = {}
        for item in self.item_catalog:
            item_lower = item.lower()
            assigned = False
            for category, keywords in category_keywords.items():
                if any(kw in item_lower for kw in keywords):
                    cat_map[item] = category
                    assigned = True
                    break
            if not assigned:
                cat_map[item] = 'other'

        return cat_map

    def _normalize_prices(self) -> dict:
        """
        Log-scaled min-max normalization of item prices.
        Gently pushes toward higher-value items without spamming expensive ones.
        """
        if not self.item_avg_price:
            return {}

        prices = {k: max(v, 1.0) for k, v in self.item_avg_price.items()}
        log_prices = {k: np.log1p(v) for k, v in prices.items()}

        min_lp = min(log_prices.values())
        max_lp = max(log_prices.values())
        range_lp = max_lp - min_lp if max_lp > min_lp else 1.0

        return {
            item: round((lp - min_lp) / range_lp, 4)
            for item, lp in log_prices.items()
        }

    def recommend(self, cart_items: list, context: dict = None,
                  top_k: int = 5, exclude_cart: bool = True,
                  order_metadata: dict = None) -> list:
        """
        Full 4-stage recommendation pipeline:
          Stage 1: Candidate generation (Graph + Context + BERT)  → Top 20
          Stage 2: MMR diversity filter                           → Re-ranked
          Stage 3: EV price boost                                 → Score adjusted
          Stage 4: XGBoost LTR re-rank (if available)             → Final Top K

        Args:
            cart_items: List of item names currently in cart
            context: Optional dict with keys 'daypart', 'subzone', 'price_tier'
            top_k: Number of recommendations to return
            exclude_cart: Whether to exclude items already in cart
            order_metadata: Optional dict with 'cart_value', 'is_weekend', 'kpt_duration'

        Returns:
            List of dicts: [{item, score, confidence, lift, source_graph}, ...]
        """
        start_time = time.perf_counter()

        # ================================================================
        # STAGE 1: Candidate Generation (Graph + Context + BERT)
        # ================================================================
        candidate_scores = defaultdict(lambda: {
            'score': 0, 'confidence': 0, 'lift': 0,
            'contributing_items': [], 'sources': set()
        })

        # --- Layer 1: Global Graph ---
        self._aggregate_from_graph(
            cart_items, self.global_graph, candidate_scores, 'global', weight=1.0
        )

        # --- Layer 2: Contextual Sub-Graphs (boosted weight) ---
        if context:
            ctx_keys = self._resolve_context_keys(context)
            for ctx_key in ctx_keys:
                if ctx_key in self.sub_graphs:
                    self._aggregate_from_graph(
                        cart_items, self.sub_graphs[ctx_key],
                        candidate_scores, ctx_key, weight=1.5
                    )

        # --- Layer 3: BERT Semantic Similarity Boost ---
        if self.semantic_encoder:
            self._apply_semantic_boost(cart_items, candidate_scores)

        # --- Cold Start Fallback ---
        if not candidate_scores:
            candidate_scores = self._cold_start_fallback(cart_items)

        # Filter out items already in cart
        if exclude_cart:
            cart_set = set(cart_items)
            candidate_scores = {
                k: v for k, v in candidate_scores.items()
                if k not in cart_set
            }

        # Get top-20 candidates for re-ranking stages
        top_n_for_rerank = 20
        pre_ranked = sorted(
            candidate_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_n_for_rerank]

        # ================================================================
        # STAGE 2: MMR Diversity Filter
        # ================================================================
        if self.semantic_encoder and len(pre_ranked) > 1:
            pre_ranked = self._apply_mmr(
                pre_ranked, cart_items, top_k=min(top_n_for_rerank, len(pre_ranked))
            )

        # ================================================================
        # STAGE 3: Expected Value (EV) Price Re-Ranking
        # ================================================================
        if self.normalized_prices:
            pre_ranked = self._apply_ev_reranking(pre_ranked)

        # ================================================================
        # STAGE 4: XGBoost LTR Re-Ranker (if available)
        # ================================================================
        if self.xgb_reranker and order_metadata:
            pre_ranked = self._apply_xgb_rerank(
                pre_ranked, cart_items, context or {}, order_metadata
            )

        # Final top-K
        final = pre_ranked[:top_k]
        latency_ms = (time.perf_counter() - start_time) * 1000

        results = []
        for item, data in final:
            rec = {
                'item': item,
                'score': round(data['score'], 4),
                'confidence': round(data['confidence'], 4),
                'lift': round(data['lift'], 4),
                'contributing_items': data['contributing_items'],
                'sources': list(data['sources']),
            }
            if item in self.item_avg_price:
                rec['est_price'] = self.item_avg_price[item]
            if item in self.normalized_prices:
                rec['ev_score'] = round(
                    data['score'] * (1 + self.ev_weight * self.normalized_prices[item]), 4
                )
            results.append(rec)

        return results, latency_ms

    # ------------------------------------------------------------------
    # STAGE 2: Maximal Marginal Relevance (MMR)
    # ------------------------------------------------------------------
    def _apply_mmr(self, candidates: list, cart_items: list,
                   top_k: int = 20) -> list:
        """
        Maximal Marginal Relevance re-ranking for diversity.

        MMR(i) = λ · Relevance(i, Cart) − (1−λ) · max_{j∈S} Sim(i, j)

        Balances relevance (how well item fits the cart) with diversity
        (how different it is from already selected recommendations).
        """
        if not candidates:
            return candidates

        lam = self.mmr_lambda
        remaining = list(candidates)
        selected = []

        # Normalize scores to [0,1] for MMR
        max_score = max(d['score'] for _, d in remaining) if remaining else 1.0
        max_score = max(max_score, 1e-10)

        # Select iteratively
        while remaining and len(selected) < top_k:
            best_mmr = -float('inf')
            best_idx = 0

            for idx, (item, data) in enumerate(remaining):
                # Relevance: normalized candidate score
                relevance = data['score'] / max_score

                # Diversity penalty: max similarity to already selected items
                if selected:
                    max_sim_to_selected = max(
                        self.semantic_encoder.get_similarity(item, sel_item)
                        for sel_item, _ in selected
                    )
                else:
                    max_sim_to_selected = 0.0

                mmr = lam * relevance - (1 - lam) * max_sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            # Move best candidate from remaining to selected
            chosen = remaining.pop(best_idx)
            chosen[1]['sources'].add('mmr_diversity')
            selected.append(chosen)

        return selected

    # ------------------------------------------------------------------
    # STAGE 3: Expected Value Price Re-Ranking
    # ------------------------------------------------------------------
    def _apply_ev_reranking(self, candidates: list) -> list:
        """
        Expected Value re-ranking:
        EV = P(acceptance) × log_normalized_price

        Gently boosts higher-value add-ons without spamming expensive items.
        Uses log-scaled normalization to prevent price domination.
        """
        for item, data in candidates:
            norm_price = self.normalized_prices.get(item, 0.5)  # Default mid-range
            # EV boost: multiplicative, controlled by ev_weight
            ev_multiplier = 1.0 + self.ev_weight * norm_price
            data['score'] *= ev_multiplier
            data['sources'].add('ev_price_boost')

        # Re-sort by EV-adjusted score
        candidates.sort(key=lambda x: x[1]['score'], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # STAGE 4: XGBoost LTR Re-Ranker
    # ------------------------------------------------------------------
    def _apply_xgb_rerank(self, candidates: list, cart_items: list,
                           context: dict, order_metadata: dict) -> list:
        """
        XGBoost Learning-to-Rank re-ranker.
        Extracts tabular features for top-20 candidates and re-ranks
        using a pre-trained XGBClassifier.
        """
        if not self.xgb_reranker or not candidates:
            return candidates

        features = []
        for item, data in candidates:
            feat = self.xgb_reranker.extract_features(
                candidate_item=item,
                candidate_data=data,
                cart_items=cart_items,
                context=context,
                order_metadata=order_metadata,
                semantic_encoder=self.semantic_encoder,
                item_avg_price=self.item_avg_price,
            )
            features.append(feat)

        # Predict acceptance probability
        import pandas as pd
        feature_df = pd.DataFrame(features)
        probs = self.xgb_reranker.predict(feature_df)

        # Blend XGBoost probability with existing score (70/30 split)
        for idx, (item, data) in enumerate(candidates):
            xgb_prob = float(probs[idx])
            blended = 0.7 * xgb_prob + 0.3 * (data['score'] / max(d['score'] for _, d in candidates))
            data['score'] = round(blended, 4)
            data['xgb_prob'] = round(xgb_prob, 4)
            data['sources'].add('xgb_rerank')

        candidates.sort(key=lambda x: x[1]['score'], reverse=True)
        return candidates

    def _aggregate_from_graph(self, cart_items: list, graph: dict,
                               scores: dict, source: str, weight: float = 1.0):
        """Aggregate outgoing edge scores from a graph for all cart items."""
        for cart_item in cart_items:
            if cart_item in graph:
                for edge in graph[cart_item]:
                    target = edge['item']
                    scores[target]['score'] += edge['score'] * weight
                    scores[target]['confidence'] += edge['confidence'] * weight
                    scores[target]['lift'] += edge['lift'] * weight
                    scores[target]['contributing_items'].append(cart_item)
                    scores[target]['sources'].add(source)

    def _resolve_context_keys(self, context: dict) -> list:
        """Resolve context dict to sub-graph keys."""
        keys = []
        if 'daypart' in context:
            keys.append(f"temporal_{context['daypart']}")
        if 'subzone' in context:
            keys.append(f"spatial_{context['subzone'].replace(' ', '_').lower()}")
        if 'price_tier' in context:
            keys.append(f"monetary_{context['price_tier']}")
        return keys

    def _apply_semantic_boost(self, cart_items: list, candidate_scores: dict):
        """
        BERT Semantic Similarity Layer:
        For each candidate already scored by the graph, boost its score
        if it is semantically related to any cart item.
        This helps surface contextually relevant items even if their
        pure co-occurrence stats are weaker.
        """
        if not self.semantic_encoder:
            return

        for target_item in list(candidate_scores.keys()):
            max_sem_sim = 0
            for cart_item in cart_items:
                sim = self.semantic_encoder.get_similarity(cart_item, target_item)
                max_sem_sim = max(max_sem_sim, sim)

            # Apply semantic boost: items with high semantic similarity get a bonus
            # but we keep it multiplicative to not override co-occurrence signals
            if max_sem_sim > 0.3:  # Only boost if meaningfully similar
                sem_boost = 1.0 + (max_sem_sim - 0.3) * 0.5  # 1.0 to 1.35 range
                candidate_scores[target_item]['score'] *= sem_boost
                candidate_scores[target_item]['sources'].add('bert_semantic_boost')

    def _cold_start_fallback(self, cart_items: list) -> dict:
        """
        Fallback for items with no graph edges.
        Strategy:
          1. (Primary) BERT nearest-neighbor: find most similar catalog items,
             inherit their graph edges weighted by cosine similarity.
          2. (Fallback) Keyword category matching if BERT unavailable.
        """
        candidate_scores = defaultdict(lambda: {
            'score': 0, 'confidence': 0, 'lift': 0,
            'contributing_items': [], 'sources': set()
        })

        for cart_item in cart_items:
            # --- Strategy 1: BERT nearest-neighbor ---
            if self.semantic_encoder:
                neighbors = self.semantic_encoder.get_nearest_neighbors(
                    cart_item, top_k=5, exclude_self=True
                )
                for neighbor_item, sim_score in neighbors:
                    if neighbor_item in self.global_graph and sim_score > 0.3:
                        # Weight inheritance by semantic similarity
                        weight = sim_score * 0.7  # Scale: 0.21 to 0.7
                        for edge in self.global_graph[neighbor_item][:5]:
                            target = edge['item']
                            candidate_scores[target]['score'] += edge['score'] * weight
                            candidate_scores[target]['confidence'] += edge['confidence'] * weight
                            candidate_scores[target]['lift'] += edge['lift'] * weight
                            candidate_scores[target]['contributing_items'].append(
                                f"{cart_item} →BERT({sim_score:.2f})→ {neighbor_item}"
                            )
                            candidate_scores[target]['sources'].add('bert_cold_start')
            else:
                # --- Strategy 2: Keyword category fallback ---
                cart_category = self.category_map.get(cart_item, 'other')
                for catalog_item in self.item_catalog:
                    if (self.category_map.get(catalog_item) == cart_category and
                        catalog_item in self.global_graph and
                        catalog_item != cart_item):
                        for edge in self.global_graph[catalog_item][:3]:
                            target = edge['item']
                            candidate_scores[target]['score'] += edge['score'] * 0.5
                            candidate_scores[target]['confidence'] += edge['confidence'] * 0.5
                            candidate_scores[target]['lift'] += edge['lift'] * 0.5
                            candidate_scores[target]['contributing_items'].append(
                                f"{cart_item} (via {catalog_item})"
                            )
                            candidate_scores[target]['sources'].add('keyword_cold_start')

        return candidate_scores


# ============================================================================
# COMPONENT 7: SASREC TRANSFORMER (OFFLINE FEATURE EXTRACTOR)
# ============================================================================

class SASRecModel:
    """
    Self-Attentive Sequential Recommendation (SASRec) Transformer.
    Used OFFLINE to learn item-item sequential transition patterns.

    Architecture:
      - Item embedding layer (64-dim)
      - Positional encoding
      - 2-layer Transformer encoder (2 heads)
      - Trained on next-item prediction over order sequences

    Output: Pre-computed transition score matrix for O(1) lookup at inference.
    Zero online latency cost — scores are pre-materialized.
    """

    def __init__(self, item_catalog: set, embed_dim: int = 64,
                 n_heads: int = 2, n_layers: int = 2, max_seq_len: int = 20):
        import torch
        import torch.nn as nn

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Build item-to-index mapping
        self.items = sorted(list(item_catalog))
        self.item2idx = {item: idx + 1 for idx, item in enumerate(self.items)}  # 0 = padding
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.n_items = len(self.items) + 1  # +1 for padding token

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Will be populated after training
        self.transition_matrix = {}  # {source_item: {target_item: score}}

        print(f"\n=== SASREC TRANSFORMER ===")
        print(f"  Items: {len(self.items)}, Embed dim: {embed_dim}, "
              f"Layers: {n_layers}, Heads: {n_heads}")
        print(f"  Device: {self.device}")

    def _build_model(self):
        """Build the SASRec Transformer model."""
        import torch
        import torch.nn as nn

        class SASRec(nn.Module):
            def __init__(self, n_items, embed_dim, n_heads, n_layers, max_seq_len):
                super().__init__()
                self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
                self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
                self.dropout = nn.Dropout(0.2)
                self.layer_norm = nn.LayerNorm(embed_dim)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.2,
                    activation='gelu',
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )

                self.output_layer = nn.Linear(embed_dim, n_items)

            def forward(self, item_seq, padding_mask=None):
                seq_len = item_seq.size(1)
                positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)

                x = self.item_embedding(item_seq) + self.pos_embedding(positions)
                x = self.layer_norm(self.dropout(x))

                # Causal mask: each position can only attend to previous positions
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1
                ).bool()

                x = self.transformer(x, mask=causal_mask,
                                     src_key_padding_mask=padding_mask)

                logits = self.output_layer(x)  # (batch, seq_len, n_items)
                return logits

        return SASRec(self.n_items, self.embed_dim, self.n_heads,
                      self.n_layers, self.max_seq_len)

    def _prepare_sequences(self, orders: list) -> list:
        """Convert orders into padded item index sequences."""
        sequences = []
        for order in orders:
            items = order.get('unique_items', [])
            if len(items) < 2:
                continue
            # Convert to indices
            seq = [self.item2idx.get(item, 0) for item in items]
            # Truncate to max_seq_len
            seq = seq[:self.max_seq_len]
            sequences.append(seq)
        return sequences

    def train_model(self, orders: list, epochs: int = 15, batch_size: int = 128,
                    lr: float = 0.001):
        """
        Train SASRec on order sequences using next-item prediction.
        For sequence [A, B, C, D]:
          Input:  [A, B, C]  →  Target: [B, C, D]
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        sequences = self._prepare_sequences(orders)
        print(f"  Training sequences: {len(sequences)}")

        if not sequences:
            print("  WARNING: No valid sequences for SASRec training")
            return

        # Pad sequences
        max_len = min(self.max_seq_len, max(len(s) for s in sequences))
        padded_input = []
        padded_target = []
        padding_masks = []

        for seq in sequences:
            # Input: all but last, Target: all but first
            inp = seq[:-1]
            tgt = seq[1:]
            seq_len = len(inp)

            # Pad to max_len - 1
            pad_len = max(0, max_len - 1 - seq_len)
            inp_padded = [0] * pad_len + inp
            tgt_padded = [0] * pad_len + tgt
            mask = [True] * pad_len + [False] * seq_len

            # Truncate if needed
            inp_padded = inp_padded[-(max_len - 1):]
            tgt_padded = tgt_padded[-(max_len - 1):]
            mask = mask[-(max_len - 1):]

            padded_input.append(inp_padded)
            padded_target.append(tgt_padded)
            padding_masks.append(mask)

        X = torch.tensor(padded_input, dtype=torch.long)
        Y = torch.tensor(padded_target, dtype=torch.long)
        M = torch.tensor(padding_masks, dtype=torch.bool)

        dataset = TensorDataset(X, Y, M)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model.train()
        print(f"  Training for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for batch_x, batch_y, batch_m in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_m = batch_m.to(self.device)

                logits = self.model(batch_x, padding_mask=batch_m)
                # logits: (batch, seq_len, n_items)
                # Reshape for cross-entropy
                loss = criterion(
                    logits.reshape(-1, self.n_items),
                    batch_y.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        print(f"  Training complete. Final loss: {avg_loss:.4f}")

    def compute_transition_matrix(self):
        """
        Pre-compute item-item transition scores from the trained model.
        For each item, predict the probability of every other item following it.
        This matrix is exported as JSON for O(1) lookup at inference time.
        """
        import torch

        print("  Computing transition matrix...")
        self.model.eval()

        self.transition_matrix = {}

        with torch.no_grad():
            for item_name, item_idx in self.item2idx.items():
                # Single-item sequence: what follows this item?
                seq = torch.tensor([[item_idx]], dtype=torch.long).to(self.device)
                logits = self.model(seq)  # (1, 1, n_items)

                # Convert to probabilities
                probs = torch.softmax(logits[0, -1, :], dim=0).cpu().numpy()

                # Store top-20 transitions (sparse representation)
                top_indices = np.argsort(probs)[::-1][:20]
                transitions = {}
                for idx in top_indices:
                    idx = int(idx)
                    if idx in self.idx2item and idx != 0:
                        transitions[self.idx2item[idx]] = round(float(probs[idx]), 6)

                self.transition_matrix[item_name] = transitions

        print(f"  Transition matrix: {len(self.transition_matrix)} items, "
              f"top-20 targets each")

    def get_transition_score(self, source_item: str, target_item: str) -> float:
        """O(1) lookup: P(target | source) from pre-computed matrix."""
        if source_item in self.transition_matrix:
            return self.transition_matrix[source_item].get(target_item, 0.0)
        return 0.0

    def get_max_transition_score(self, cart_items: list, candidate: str) -> float:
        """Max transition score from any cart item to the candidate."""
        max_score = 0.0
        for cart_item in cart_items:
            score = self.get_transition_score(cart_item, candidate)
            max_score = max(max_score, score)
        return max_score

    def export_transitions(self, path: str):
        """Export transition matrix as JSON."""
        export_data = {
            'model': 'SASRec',
            'embed_dim': self.embed_dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'num_items': len(self.items),
            'top_transitions_per_item': 20,
            'sample_transitions': {
                item: self.transition_matrix.get(item, {})
                for item in self.items[:15]  # Sample for inspection
            },
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"  -> SASRec transitions saved: {path}")


# ============================================================================
# COMPONENT 6: XGBOOST LEARNING-TO-RANK RE-RANKER
# ============================================================================

class XGBoostReRanker:
    """
    Lightweight XGBoost Learning-to-Rank model.
    Trained on historical order data to predict P(item_added | cart_state).

    Features per candidate:
      1. graph_confidence   - Association rule confidence score
      2. bert_similarity    - Max BERT cosine similarity to any cart item
      3. item_price_norm    - Log-scaled normalized price
      4. kpt_duration       - Avg KPT (kitchen prep time) for this item
      5. is_weekend         - Whether the order is on a weekend
      6. cart_value          - Current cart subtotal (normalized)
      7. sasrec_score       - SASRec Transformer transition probability
    """

    FEATURE_NAMES = [
        'graph_confidence', 'bert_similarity', 'item_price_norm',
        'kpt_duration', 'is_weekend', 'cart_value_norm', 'sasrec_score'
    ]

    def __init__(self, sasrec_model: SASRecModel = None):
        self.model = None
        self.is_trained = False
        self.sasrec_model = sasrec_model

    def generate_training_data(self, orders: list, global_graph: dict,
                                semantic_encoder: SemanticEncoder,
                                item_avg_price: dict,
                                sample_size: int = 2000,
                                sasrec_model: SASRecModel = None) -> tuple:
        """
        Generate training data using leave-one-out from historical orders.
        For each order with ≥3 items:
          - Hold out each item as a positive example (label=1)
          - Sample random non-order items as negatives (label=0)
        """
        import random

        print("\n=== XGBOOST LTR: Generating Training Data ===")

        eligible = [o for o in orders if len(o['unique_items']) >= 3]
        sample = eligible[:min(sample_size, len(eligible))]
        all_items = list(item_avg_price.keys())

        rows = []
        labels = []

        for order in sample:
            items = order['unique_items']
            cart_value = order.get('bill_subtotal', 0)
            is_weekend = 1 if order.get('is_weekend', False) else 0

            for hold_out_idx in range(len(items)):
                held_out = items[hold_out_idx]
                cart = [items[j] for j in range(len(items)) if j != hold_out_idx]

                if not cart:
                    continue

                # --- Positive example: held-out item ---
                feat_pos = self._compute_features(
                    candidate_item=held_out,
                    cart_items=cart,
                    global_graph=global_graph,
                    semantic_encoder=semantic_encoder,
                    item_avg_price=item_avg_price,
                    cart_value=cart_value,
                    is_weekend=is_weekend,
                )
                rows.append(feat_pos)
                labels.append(1)

                # --- Negative example: random item NOT in order ---
                neg_candidates = [i for i in all_items if i not in items]
                if neg_candidates:
                    neg_item = random.choice(neg_candidates)
                    feat_neg = self._compute_features(
                        candidate_item=neg_item,
                        cart_items=cart,
                        global_graph=global_graph,
                        semantic_encoder=semantic_encoder,
                        item_avg_price=item_avg_price,
                        cart_value=cart_value,
                        is_weekend=is_weekend,
                    )
                    rows.append(feat_neg)
                    labels.append(0)

        X = pd.DataFrame(rows)
        y = np.array(labels)

        print(f"  Training samples: {len(X)} ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        return X, y

    def _compute_features(self, candidate_item: str, cart_items: list,
                           global_graph: dict, semantic_encoder: SemanticEncoder,
                           item_avg_price: dict, cart_value: float,
                           is_weekend: int) -> dict:
        """Compute feature vector for a single candidate item."""
        # 1. Graph confidence: max confidence from any cart item → candidate
        graph_conf = 0.0
        for cart_item in cart_items:
            if cart_item in global_graph:
                for edge in global_graph[cart_item]:
                    if edge['item'] == candidate_item:
                        graph_conf = max(graph_conf, edge['confidence'])
                        break

        # 2. BERT similarity: max cosine sim to any cart item
        bert_sim = 0.0
        if semantic_encoder:
            for cart_item in cart_items:
                sim = semantic_encoder.get_similarity(cart_item, candidate_item)
                bert_sim = max(bert_sim, sim)

        # 3. Normalized item price
        price = item_avg_price.get(candidate_item, 200)
        price_norm = np.log1p(max(price, 1)) / 10.0  # Rough normalization

        # 4. Cart value (normalized by 1000)
        cart_value_norm = cart_value / 1000.0

        # 5. SASRec transition score
        sasrec_score = 0.0
        if self.sasrec_model:
            sasrec_score = self.sasrec_model.get_max_transition_score(
                cart_items, candidate_item
            )

        return {
            'graph_confidence': round(graph_conf, 4),
            'bert_similarity': round(bert_sim, 4),
            'item_price_norm': round(price_norm, 4),
            'kpt_duration': 0.0,  # Placeholder: can be enriched with real data
            'is_weekend': is_weekend,
            'cart_value_norm': round(cart_value_norm, 4),
            'sasrec_score': round(sasrec_score, 6),
        }

    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Train XGBClassifier for P(added | features)."""
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        print("  Training XGBClassifier...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            random_state=42,
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Print validation metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        val_preds = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_preds)
        val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))

        print(f"  Validation AUC: {val_auc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")

        # Feature importances
        importances = self.model.feature_importances_
        print(f"  Feature Importances:")
        for fname, imp in sorted(zip(self.FEATURE_NAMES, importances),
                                   key=lambda x: x[1], reverse=True):
            print(f"    {fname}: {imp:.4f}")

        self.is_trained = True
        self.val_auc = val_auc
        self.val_acc = val_acc

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predict acceptance probability for candidate items."""
        if not self.is_trained:
            return np.ones(len(feature_df)) * 0.5  # Fallback
        return self.model.predict_proba(feature_df)[:, 1]

    def extract_features(self, candidate_item: str, candidate_data: dict,
                          cart_items: list, context: dict,
                          order_metadata: dict,
                          semantic_encoder: SemanticEncoder,
                          item_avg_price: dict) -> dict:
        """Extract features for a single candidate during live inference."""
        # Graph confidence from pre-aggregated data
        graph_conf = candidate_data.get('confidence', 0)

        # BERT similarity
        bert_sim = 0.0
        if semantic_encoder:
            for cart_item in cart_items:
                sim = semantic_encoder.get_similarity(cart_item, candidate_item)
                bert_sim = max(bert_sim, sim)

        # Price
        price = item_avg_price.get(candidate_item, 200)
        price_norm = np.log1p(max(price, 1)) / 10.0

        # Order metadata
        cart_value = order_metadata.get('cart_value', 0) / 1000.0
        is_weekend = 1 if order_metadata.get('is_weekend', False) else 0

        # SASRec transition score
        sasrec_score = 0.0
        if self.sasrec_model:
            sasrec_score = self.sasrec_model.get_max_transition_score(
                cart_items, candidate_item
            )

        return {
            'graph_confidence': round(graph_conf, 4),
            'bert_similarity': round(bert_sim, 4),
            'item_price_norm': round(price_norm, 4),
            'kpt_duration': 0.0,
            'is_weekend': is_weekend,
            'cart_value_norm': round(cart_value, 4),
            'sasrec_score': round(sasrec_score, 6),
        }

    def save_model(self, path: str):
        """Save trained model."""
        if self.is_trained:
            self.model.save_model(path)
            print(f"  -> XGBoost model saved: {path}")


# ============================================================================
# COMPONENT 5: SERIALIZATION & EVALUATION
# ============================================================================

class PipelineSerializer:
    """Serializes computed graphs to JSON for Redis/in-memory cache deployment."""

    @staticmethod
    def serialize_to_json(global_graph: dict, sub_graphs: dict,
                          item_catalog: set, output_dir: str = '.'):
        """Export all graphs as JSON files."""
        import os

        # Global graph
        global_path = os.path.join(output_dir, 'csao_global_graph.json')
        with open(global_path, 'w') as f:
            json.dump(global_graph, f, indent=2)
        print(f"  -> Saved global graph: {global_path} ({len(global_graph)} nodes)")

        # Sub-graphs (single file)
        sub_path = os.path.join(output_dir, 'csao_sub_graphs.json')
        with open(sub_path, 'w') as f:
            json.dump(sub_graphs, f, indent=2)
        print(f"  -> Saved sub-graphs: {sub_path} ({len(sub_graphs)} contexts)")

        # Item catalog
        catalog_path = os.path.join(output_dir, 'csao_item_catalog.json')
        with open(catalog_path, 'w') as f:
            json.dump(sorted(list(item_catalog)), f, indent=2)
        print(f"  -> Saved catalog: {catalog_path} ({len(item_catalog)} items)")


class Evaluator:
    """Evaluates the recommendation system with proxy metrics."""

    def __init__(self, engine: RecommendationEngine, orders: list):
        self.engine = engine
        self.orders = orders

    def evaluate_hit_rate(self, sample_size: int = 500) -> dict:
        """
        Leave-one-out evaluation:
        For orders with ≥3 items, hold out the last item,
        use remaining as cart, check if held-out is in top-K.
        """
        print("\n=== EVALUATION: Leave-One-Out Hit Rate ===")

        eligible = [o for o in self.orders if len(o['unique_items']) >= 3]
        sample = eligible[:min(sample_size, len(eligible))]

        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        total = 0
        total_latency = 0

        for order in sample:
            items = order['unique_items']
            # Hold out last item
            cart = items[:-1]
            held_out = items[-1]

            context = {
                'daypart': order.get('daypart', 'dinner'),
                'subzone': order.get('subzone', 'Unknown'),
                'price_tier': order.get('price_tier', 'mid'),
            }

            recs, latency = self.engine.recommend(cart, context=context, top_k=10)
            total_latency += latency
            total += 1

            rec_items = [r['item'] for r in recs]
            for k in hits_at_k:
                if held_out in rec_items[:k]:
                    hits_at_k[k] += 1

        results = {}
        for k, hits in hits_at_k.items():
            rate = hits / total if total > 0 else 0
            results[f'Hit@{k}'] = round(rate, 4)
            print(f"  Hit@{k}: {rate:.2%} ({hits}/{total})")

        avg_latency = total_latency / total if total > 0 else 0
        results['avg_latency_ms'] = round(avg_latency, 3)
        print(f"  Avg Latency: {avg_latency:.3f} ms")

        return results

    def evaluate_aov_uplift(self, sample_size: int = 500) -> dict:
        """Estimate potential AOV uplift from CSAO recommendations."""
        print("\n=== EVALUATION: AOV Uplift Potential ===")

        single_item = [o for o in self.orders if len(o['unique_items']) == 1]
        sample = single_item[:min(sample_size, len(single_item))]

        if not sample:
            print("  No single-item orders found for AOV analysis.")
            return {}

        # Average bill for single-item vs multi-item orders
        single_avg = np.mean([o['bill_subtotal'] for o in single_item]) if single_item else 0
        multi_item = [o for o in self.orders if len(o['unique_items']) >= 2]
        multi_avg = np.mean([o['bill_subtotal'] for o in multi_item]) if multi_item else 0

        potential_uplift = ((multi_avg - single_avg) / single_avg * 100) if single_avg > 0 else 0

        print(f"  Single-item Avg Bill: ₹{single_avg:.2f}")
        print(f"  Multi-item Avg Bill:  ₹{multi_avg:.2f}")
        print(f"  Potential AOV Uplift: {potential_uplift:.1f}%")

        # Simulate CSAO attach rate
        attach_count = 0
        for order in sample:
            context = {
                'daypart': order.get('daypart', 'dinner'),
                'subzone': order.get('subzone', 'Unknown'),
                'price_tier': order.get('price_tier', 'mid'),
            }
            recs, _ = self.engine.recommend(order['unique_items'], context=context, top_k=3)
            if recs:
                attach_count += 1

        attach_rate = attach_count / len(sample) if sample else 0
        print(f"  CSAO Rail Attach Rate: {attach_rate:.2%} ({attach_count}/{len(sample)})")

        return {
            'single_item_avg_bill': round(single_avg, 2),
            'multi_item_avg_bill': round(multi_avg, 2),
            'potential_aov_uplift_pct': round(potential_uplift, 1),
            'csao_attach_rate': round(attach_rate, 4),
        }


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def run_pipeline(csv_path: str):
    """Orchestrate the full CSAO pipeline."""

    print("=" * 70)
    print("  ZOMATHON: CSAO Rail Recommendation System Pipeline")
    print("=" * 70)
    pipeline_start = time.time()

    # ── Component 1: Data Processing ──
    processor = DataProcessor(csv_path)
    processor.load_and_clean()
    processor.parse_items()
    processor.engineer_features()
    multi_item_orders = processor.get_multi_item_orders()

    # ── Component 2: Global Association Rule Mining ──
    engine = AssociationRuleEngine(
        orders=processor.parsed_orders,
        item_catalog=processor.item_catalog,
        min_support=0.0005,   # Relaxed for ~21K orders
        min_confidence=0.03,
        min_lift=1.0,
    )
    engine.compute_support()
    engine.compute_pair_metrics()
    engine.filter_rules()
    global_graph = engine.build_graph()

    # ── Component 3: Contextual Sub-Graphs ──
    ctx_builder = ContextualGraphBuilder(
        orders=processor.parsed_orders,
        item_catalog=processor.item_catalog,
        min_support=0.0005,
        min_confidence=0.03,
        min_lift=1.0,
    )
    sub_graphs = ctx_builder.build_all_sub_graphs()

    # ── Component 3.5: BERT Semantic Encoding ──
    semantic_encoder = SemanticEncoder(processor.item_catalog)

    # ── Component 7: SASRec Transformer (Offline Feature Extractor) ──
    sasrec = SASRecModel(
        item_catalog=processor.item_catalog,
        embed_dim=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=20,
    )
    sasrec.train_model(processor.parsed_orders, epochs=15, batch_size=128)
    sasrec.compute_transition_matrix()

    # ── Component 6: XGBoost LTR Re-Ranker (now with SASRec feature) ──
    xgb_reranker = XGBoostReRanker(sasrec_model=sasrec)
    X_train, y_train = xgb_reranker.generate_training_data(
        orders=processor.parsed_orders,
        global_graph=global_graph,
        semantic_encoder=semantic_encoder,
        item_avg_price=processor.item_avg_price,
        sample_size=2000,
        sasrec_model=sasrec,
    )
    xgb_reranker.train(X_train, y_train)

    # ── Component 4: Initialize Recommendation Engine ──
    rec_engine = RecommendationEngine(
        global_graph=global_graph,
        sub_graphs=sub_graphs,
        item_catalog=processor.item_catalog,
        semantic_encoder=semantic_encoder,
        item_avg_price=processor.item_avg_price,
        xgb_reranker=xgb_reranker,
        mmr_lambda=0.6,
        ev_weight=0.3,
    )

    # ── Component 5: Serialize & Evaluate ──
    import os
    output_dir = os.path.dirname(csv_path)

    print("\n=== SERIALIZATION ===")
    PipelineSerializer.serialize_to_json(
        global_graph, sub_graphs, processor.item_catalog, output_dir
    )
    semantic_encoder.export_embeddings(
        os.path.join(output_dir, 'csao_semantic_embeddings.json')
    )
    sasrec.export_transitions(
        os.path.join(output_dir, 'csao_sasrec_transitions.json')
    )
    xgb_reranker.save_model(
        os.path.join(output_dir, 'csao_xgb_reranker.json')
    )

    evaluator = Evaluator(rec_engine, processor.parsed_orders)
    hit_results = evaluator.evaluate_hit_rate()
    aov_results = evaluator.evaluate_aov_uplift()

    # ── Interactive Demo ──
    print("\n" + "=" * 70)
    print("  LIVE DEMO: CSAO Recommendations")
    print("=" * 70)

    demo_carts = [
        {
            'cart': ['Bone in Jamaican Grilled Chicken'],
            'context': {'daypart': 'dinner', 'price_tier': 'mid'},
            'label': 'Single chicken item at dinner'
        },
        {
            'cart': ['Bageecha Pizza', 'Cheesy Garlic Bread'],
            'context': {'daypart': 'dinner', 'price_tier': 'premium'},
            'label': 'Pizza + Bread combo (incomplete meal → beverage/side expected)'
        },
        {
            'cart': ['Animal Fries'],
            'context': {'daypart': 'late_night', 'price_tier': 'budget'},
            'label': 'Late-night snack, budget cart'
        },
        {
            'cart': ['Fried Chicken Peri Peri Tender', 'Peri Peri Fries'],
            'context': {'daypart': 'lunch', 'price_tier': 'mid'},
            'label': 'Peri Peri combo → sauce/drink completion'
        },
        {
            'cart': ['Margherita Pizza'],
            'context': {'daypart': 'dinner', 'price_tier': 'mid'},
            'label': 'Single pizza → garlic bread/sides expected'
        },
    ]

    for demo in demo_carts:
        print(f"\n🛒 Cart: {demo['cart']}")
        print(f"   Context: {demo['context']}")
        print(f"   Scenario: {demo['label']}")

        order_meta = {
            'cart_value': sum(processor.item_avg_price.get(i, 200) for i in demo['cart']),
            'is_weekend': False,
            'kpt_duration': 30,
        }

        recs, latency = rec_engine.recommend(
            demo['cart'],
            context=demo['context'],
            top_k=5,
            order_metadata=order_meta,
        )

        if recs:
            print(f"   ⚡ Recommendations ({latency:.2f}ms):")
            for i, r in enumerate(recs, 1):
                price_str = f", ₹{r['est_price']:.0f}" if 'est_price' in r else ''
                print(f"      {i}. {r['item']}  "
                      f"[Score={r['score']}, Conf={r['confidence']}{price_str}]")
                print(f"         Sources: {', '.join(r['sources'])}")
        else:
            print(f"   ❌ No recommendations (cold start scenario)")

    # ── Latency Benchmark ──
    print("\n=== LATENCY BENCHMARK ===")
    latencies = []
    for _ in range(1000):
        cart = ['Bone in Jamaican Grilled Chicken', 'Animal Fries']
        context = {'daypart': 'dinner', 'price_tier': 'mid'}
        order_meta = {'cart_value': 500, 'is_weekend': False, 'kpt_duration': 30}
        _, lat = rec_engine.recommend(cart, context=context, top_k=5, order_metadata=order_meta)
        latencies.append(lat)

    print(f"  1000-iteration benchmark:")
    print(f"  P50 Latency: {np.percentile(latencies, 50):.3f} ms")
    print(f"  P95 Latency: {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99 Latency: {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max Latency: {max(latencies):.3f} ms")

    pipeline_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  Pipeline completed in {pipeline_time:.1f}s")
    print(f"{'='*70}")

    # Save full evaluation results
    eval_results = {
        'hit_rate': hit_results,
        'aov_uplift': aov_results,
        'latency_benchmark': {
            'p50_ms': round(np.percentile(latencies, 50), 3),
            'p95_ms': round(np.percentile(latencies, 95), 3),
            'p99_ms': round(np.percentile(latencies, 99), 3),
            'max_ms': round(max(latencies), 3),
        },
        'graph_stats': {
            'total_items': len(processor.item_catalog),
            'total_orders': len(processor.parsed_orders),
            'multi_item_orders': len(multi_item_orders),
            'global_graph_nodes': len(global_graph),
            'total_rules': len(engine.rules),
            'sub_graphs_count': len(sub_graphs),
            'bert_model': SemanticEncoder.MODEL_NAME,
            'embedding_dim': int(semantic_encoder.embeddings.shape[1]),
            'semantic_clusters': len(semantic_encoder.clusters),
        },
        'phase2_features': {
            'mmr_lambda': 0.6,
            'ev_weight': 0.3,
            'xgb_val_auc': round(xgb_reranker.val_auc, 4) if xgb_reranker.is_trained else None,
            'xgb_val_accuracy': round(xgb_reranker.val_acc, 4) if xgb_reranker.is_trained else None,
        }
    }

    eval_path = os.path.join(output_dir, 'csao_evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Evaluation results saved: {eval_path}")

    return rec_engine, processor, eval_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys
    import os

    # Default to the CSV in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, 'order_history_kaggle_data (1).csv')

    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    rec_engine, processor, eval_results = run_pipeline(csv_path)
