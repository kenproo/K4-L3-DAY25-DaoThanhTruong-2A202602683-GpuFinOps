import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_cache_break_even_and_worth_it():
    # Break-even for large tier ($3.00/1M input, $3.75 write cost, 0.10 read discount)
    be = pricing.cache_break_even_reads(write_cost_per_m=3.75, base_input_price=3.00, read_discount=0.10)
    assert abs(be - 1.388888) < 1e-4

    # If expected reads = 1 (< 1.39), caching is not worth it
    assert not pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, read_discount=0.10, base_input_price=3.00)

    # If expected reads = 3 (> 1.39), caching is worth it
    assert pricing.cache_is_worth_it(avg_cache_reads=3.0, write_cost_per_m=3.75, read_discount=0.10, base_input_price=3.00)


def test_mbu_rightsizing_extension():
    res = m1_efficiency_audit.run(verbose=False)
    assert "mbu_rightsizing" in res
    assert len(res["mbu_rightsizing"]) > 0
    assert res["monthly_rightsize_savings"] > 0


def test_carbon_aware_scheduling_extension():
    res = m3_purchasing.run(verbose=False)
    assert "carbon_scheduling" in res
    assert res["carbon_scheduling"]["cleanest_region"] == "europe-north1"
    assert res["carbon_scheduling"]["cheapest_region"] == "us-east-wa"
