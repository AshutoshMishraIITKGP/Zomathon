"""
Zomathon CSAO — Interactive Recommendation Simulations
Runs the full pipeline and displays detailed recommendations for diverse cart scenarios.
"""

import sys
import os

# Run the pipeline to build all components
print("=" * 70)
print("  LOADING PIPELINE...")
print("=" * 70)

from csao_pipeline import run_pipeline

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'order_history_kaggle_data (1).csv')
rec_engine, processor, eval_results = run_pipeline(csv_path)

# Define simulation scenarios
scenarios = [
    {
        'cart': ['Bone in Jamaican Grilled Chicken'],
        'context': {'daypart': 'dinner', 'price_tier': 'premium'},
        'label': '🍗 Solo Chicken dinner — expecting sides, drinks, dessert'
    },
    {
        'cart': ['Bageecha Pizza', 'Cheesy Garlic Bread'],
        'context': {'daypart': 'dinner', 'price_tier': 'premium'},
        'label': '🍕 Pizza + Bread — incomplete meal, needs a beverage'
    },
    {
        'cart': ['Animal Fries'],
        'context': {'daypart': 'late_night', 'price_tier': 'budget'},
        'label': '🍟 Late-night fries — budget snacker'
    },
    {
        'cart': ['Fried Chicken Peri Peri Tender', 'Peri Peri Fries'],
        'context': {'daypart': 'lunch', 'price_tier': 'mid'},
        'label': '🔥 Peri Peri combo — tests MMR diversity (avoid more fries!)'
    },
    {
        'cart': ['Angara Rice'],
        'context': {'daypart': 'dinner', 'price_tier': 'mid'},
        'label': '🍚 Solo rice item — should suggest curries/gravies'
    },
    {
        'cart': ['Butter Chicken Boneless', 'Butter Naan'],
        'context': {'daypart': 'dinner', 'price_tier': 'premium'},
        'label': '🧈 Classic North Indian combo — dessert or drink expected'
    },
    {
        'cart': ['Coke'],
        'context': {'daypart': 'afternoon', 'price_tier': 'budget'},
        'label': '🥤 Just a Coke — cold start test, can we upsell?'
    },
    {
        'cart': ['Mexican Grilled Veg Burger', 'French Fries'],
        'context': {'daypart': 'lunch', 'price_tier': 'mid'},
        'label': '🍔 Burger + Fries — classic combo, needs a drink to complete'
    },
    {
        'cart': ['Pasta Italiano White Sauce'],
        'context': {'daypart': 'dinner', 'price_tier': 'premium'},
        'label': '🍝 Solo pasta — expecting bread/soup/drink'
    },
    {
        'cart': ['Farm House', 'Pepsi'],
        'context': {'daypart': 'dinner', 'price_tier': 'mid'},
        'label': '🏠 Farm House Pizza + Pepsi — needs a side/dessert'
    },
]


# Run simulations
print("\n" + "=" * 70)
print("  CSAO RECOMMENDATION SIMULATIONS")
print("  4-Stage Pipeline: Graph → MMR → EV Boost → XGBoost Re-Rank")
print("=" * 70)

for i, scenario in enumerate(scenarios, 1):
    cart = scenario['cart']
    context = scenario['context']
    label = scenario['label']

    print(f"\n{'─' * 70}")
    print(f"  SCENARIO {i}: {label}")
    print(f"  Cart:    {' + '.join(cart)}")
    print(f"  Context: {context.get('daypart', 'any')} | {context.get('price_tier', 'any')}")
    print(f"{'─' * 70}")

    # Run recommendation
    order_metadata = {
        'cart_value': sum(processor.item_avg_price.get(item, 200) for item in cart),
        'is_weekend': False,
    }

    results, latency_ms = rec_engine.recommend(
        cart_items=cart,
        context=context,
        top_k=5,
        order_metadata=order_metadata,
    )

    if results:
        print(f"  {'Rank':<5} {'Recommended Item':<40} {'Score':<8} {'Price':<10} {'Source'}")
        print(f"  {'─'*5} {'─'*40} {'─'*8} {'─'*10} {'─'*20}")
        for j, rec in enumerate(results, 1):
            item = rec.get('item', '?')
            score = rec.get('score', rec.get('ev_score', 0))
            price = processor.item_avg_price.get(item, 0)
            sources = ', '.join(rec.get('sources', ['graph']))
            print(f"  {j:<5} {item:<40} {score:<8.3f} {'Rs.' + str(int(price)):<10} {sources}")
    else:
        print("  ⚠️  No recommendations generated (items may not be in graph)")

    print(f"  ⏱️  Latency: {latency_ms:.2f}ms")


# Summary
print(f"\n{'=' * 70}")
print(f"  SIMULATION COMPLETE — {len(scenarios)} scenarios tested")
print(f"  Pipeline features: Graph + BERT + MMR(λ=0.6) + EV(w=0.3) + XGBoost(7 feat)")
print(f"{'=' * 70}")
