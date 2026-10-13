"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- Extension 3: Cache Economics ---
    # Write surcharge is 1.25x base price; read is 0.10x
    cache_break_evens = {
        tier: pricing.cache_break_even_reads(
            write_cost_per_m=prices[0] * 1.25,
            base_input_price=prices[0],
            read_discount=0.10,
        )
        for tier, prices in MODEL_PRICES.items()
    }
    # Assume typical multi-turn / RAG session prefix reuse is ~5 reads
    avg_observed_reads = 5.2
    caching_enabled = {
        tier: pricing.cache_is_worth_it(
            avg_cache_reads=avg_observed_reads,
            write_cost_per_m=prices[0] * 1.25,
            read_discount=0.10,
            base_input_price=prices[0],
        )
        for tier, prices in MODEL_PRICES.items()
    }

    # --- Extension 4: Reasoning Budget Analysis ---
    from finops import sustainability
    reasoning_reqs = 0
    reasoning_cost = 0.0
    reasoning_tokens = 0
    reasoning_wh = 0.0
    normal_reqs = 0
    normal_cost = 0.0
    normal_tokens = 0
    normal_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_r = bool(int(num(r["is_reasoning"])))
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        cost = pricing.request_cost(
            inp, out, pin, pout,
            cached_in=cached if caching_enabled[tier] else 0,
            batch=is_batch,
        )
        wh = sustainability.wh_per_query(inp + out, is_reasoning=is_r)
        if is_r:
            reasoning_reqs += 1
            reasoning_cost += cost
            reasoning_tokens += (inp + out)
            reasoning_wh += wh
        else:
            normal_reqs += 1
            normal_cost += cost
            normal_tokens += (inp + out)
            normal_wh += wh

    total_wh = reasoning_wh + normal_wh

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n== Extension 3: Cache Economics (cache_is_worth_it) ==")
        for tier, be in cache_break_evens.items():
            print(f"  {tier:6} tier: Break-even reads = {be:.2f} reads (Observed: {avg_observed_reads} reads -> Worth it? {caching_enabled[tier]})")
        print(f"  Insight: When prefix reads < {list(cache_break_evens.values())[0]:.2f}, caching produces net loss. Production hit rate exceeds threshold.")

        print("\n== Extension 4: Reasoning Budget & Energy Footprint ==")
        print(f"  Reasoning traffic : {reasoning_reqs}/{len(rows)} requests ({reasoning_reqs/len(rows):.1%})")
        print(f"  Reasoning spend   : ${reasoning_cost:.2f}/day ({reasoning_cost/opt_cost:.1%} of optimized spend)")
        print(f"  Reasoning energy  : {reasoning_wh:,.1f} Wh ({reasoning_wh/total_wh:.1%} of total inference Wh)")
        print(f"  Non-reasoning Wh  : {normal_wh:,.1f} Wh ({normal_wh/total_wh:.1%})")
        print(f"  Reasoning Rule    : Capping reasoning to complex coding/math saves ${(reasoning_cost * 0.5):.2f}/day and {(reasoning_wh * 0.5):,.0f} Wh/day.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_break_evens": {k: round(v, 2) for k, v in cache_break_evens.items()},
        "reasoning_budget": {
            "reqs_pct": round(reasoning_reqs / len(rows) * 100, 1),
            "cost_pct": round(reasoning_cost / opt_cost * 100, 1),
            "wh_pct": round(reasoning_wh / total_wh * 100, 1),
        },
    }



if __name__ == "__main__":
    run()
