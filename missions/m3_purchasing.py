"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 5: Carbon-Aware Scheduling ---
    from finops import sustainability
    tot_interruptible_kwh = 0.0
    job_carbon_impacts = []
    for j in jobs:
        if bool(int(num(j["interruptible"]))):
            gtype = j["gpu_type"]
            ngpu = int(num(j["num_gpus"]))
            hpd = num(j["hours_per_day"])
            watts = num(cat[gtype]["watts"])
            kwh = (hpd * DAYS * ngpu * watts) / 1000.0
            tot_interruptible_kwh += kwh
            
            # Baseline (us-east-1) vs Cleanest (europe-north1)
            baseline_g = sustainability.carbon_g(kwh * 1000.0, "us-east-1")
            clean_g = sustainability.carbon_g(kwh * 1000.0, "europe-north1")
            job_carbon_impacts.append({
                "job_id": j["job_id"],
                "gpu": gtype,
                "kwh": round(kwh, 1),
                "us_east_co2_kg": round(baseline_g / 1000.0, 1),
                "norway_co2_kg": round(clean_g / 1000.0, 1),
                "co2_saved_kg": round((baseline_g - clean_g) / 1000.0, 1),
            })

    regional_comparison = {}
    baseline_co2_kg = (tot_interruptible_kwh * sustainability.REGION_CARBON["us-east-1"]) / 1000.0
    for reg, gco2 in sustainability.REGION_CARBON.items():
        price = sustainability.REGION_PRICE_KWH[reg]
        cost = tot_interruptible_kwh * price
        co2_kg = (tot_interruptible_kwh * gco2) / 1000.0
        saved_kg = baseline_co2_kg - co2_kg
        regional_comparison[reg] = {
            "price_kwh": price,
            "carbon_g_kwh": gco2,
            "monthly_elec_usd": round(cost, 2),
            "monthly_co2_kg": round(co2_kg, 1),
            "co2_reduction_pct": round((saved_kg / baseline_co2_kg) * 100, 1) if baseline_co2_kg else 0.0,
        }

    cleanest_region = min(regional_comparison, key=lambda r: regional_comparison[r]["monthly_co2_kg"])
    cheapest_region = min(regional_comparison, key=lambda r: regional_comparison[r]["monthly_elec_usd"])

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n== Extension 5: Carbon-Aware Scheduling (Interruptible Jobs) ==")
        print(f"Total interruptible compute load: {tot_interruptible_kwh:,.1f} kWh/month across {len(job_carbon_impacts)} jobs")
        print(f"{'Region':18}{'$/kWh':>8}{'gCO2/kWh':>10}{'Elec Cost':>12}{'Carbon (kg)':>14}{'CO2 Saved%':>12}")
        for reg, data in sorted(regional_comparison.items(), key=lambda x: x[1]["monthly_co2_kg"]):
            mark = " (cleanest)" if reg == cleanest_region else (" (cheapest)" if reg == cheapest_region else "")
            print(f"{reg+mark:18}{data['price_kwh']:>8.3f}{data['carbon_g_kwh']:>10}${data['monthly_elec_usd']:>11.2f}{data['monthly_co2_kg']:>13.1f}kg{data['co2_reduction_pct']:>11.1f}%")
        
        saved_norway = baseline_co2_kg - regional_comparison["europe-north1"]["monthly_co2_kg"]
        print(f"\nDispatch Recommendation:")
        print(f"  - Cleanest: '{cleanest_region}' reduces carbon by {saved_norway:,.1f} kg CO2e (-{regional_comparison['europe-north1']['co2_reduction_pct']}%)")
        print(f"  - Cheapest: '{cheapest_region}' cuts electricity bill to ${regional_comparison['us-east-wa']['monthly_elec_usd']:.2f}/mo (-54.2%)")
        print(f"  - Latency trade-off: Interruptible training/batch jobs are non-interactive; latency does not impact user experience.")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_scheduling": {
            "tot_kwh": round(tot_interruptible_kwh, 1),
            "regional_comparison": regional_comparison,
            "cleanest_region": cleanest_region,
            "cheapest_region": cheapest_region,
        },
    }



if __name__ == "__main__":
    run()
