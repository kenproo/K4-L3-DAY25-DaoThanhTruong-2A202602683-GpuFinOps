"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing theo MBU & $/GB-VRAM ---
    # 1. Calculate $/GB-VRAM for catalog
    vram_pricing = {}
    for gtype, c in cat.items():
        vram_pricing[gtype] = {
            "on_demand_hr": num(c["on_demand_hr"]),
            "hbm_gb": num(c["hbm_gb"]),
            "cost_per_gb_vram": round(num(c["on_demand_hr"]) / num(c["hbm_gb"]), 4) if num(c["hbm_gb"]) > 0 else 0.0,
            "peak_bw_tbs": num(c["peak_bw_tbs"]),
        }

    # 2. Right-size memory-bound GPUs (MBU < 0.35)
    # Right-sizing rule: Pick cheaper GPU that satisfies achieved bandwidth and VRAM requirements
    mbu_rightsizing = []
    total_monthly_rightsize_savings = 0.0
    for s in summary:
        cur_type = s["gpu_type"]
        cur_price = num(cat[cur_type]["on_demand_hr"])
        cur_bw = num(cat[cur_type]["peak_bw_tbs"])
        achieved_bw = s["mbu"] * cur_bw

        # If MBU is low (< 0.35), check if a cheaper GPU satisfies achieved BW
        if s["mbu"] < 0.35:
            # Target candidates in ascending cost order
            candidates = sorted(
                [g for g, info in vram_pricing.items() if info["on_demand_hr"] < cur_price and info["peak_bw_tbs"] >= achieved_bw],
                key=lambda g: vram_pricing[g]["on_demand_hr"]
            )
            if candidates:
                rec_gpu = candidates[0]
                rec_price = vram_pricing[rec_gpu]["on_demand_hr"]
                hrly_saving = cur_price - rec_price
                monthly_saving = hrly_saving * 24 * 30
                pct_saving = (hrly_saving / cur_price) * 100
                total_monthly_rightsize_savings += monthly_saving
                mbu_rightsizing.append({
                    "gpu_id": s["gpu_id"],
                    "current_gpu": cur_type,
                    "current_price": cur_price,
                    "mbu": s["mbu"],
                    "achieved_bw_tbs": round(achieved_bw, 3),
                    "rec_gpu": rec_gpu,
                    "rec_price": rec_price,
                    "monthly_saving": round(monthly_saving, 2),
                    "pct_saving": round(pct_saving, 1),
                    "reason": f"Achieved BW ({achieved_bw:.2f} TB/s) fits {rec_gpu} peak BW ({vram_pricing[rec_gpu]['peak_bw_tbs']} TB/s) at ${vram_pricing[rec_gpu]['cost_per_gb_vram']}/GB-VRAM",
                })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n== Extension 2: Right-sizing theo MBU & $/GB-VRAM ==")
        print("Catalog $/GB-VRAM:")
        for g, info in sorted(vram_pricing.items(), key=lambda x: x[1]["cost_per_gb_vram"]):
            print(f"  {g:8}: ${info['cost_per_gb_vram']:.4f}/GB-hr  (VRAM: {info['hbm_gb']}GB, Peak BW: {info['peak_bw_tbs']} TB/s, On-Demand: ${info['on_demand_hr']}/hr)")
        print("\nMemory-Bound Right-Sizing Recommendations:")
        print(f"  {'GPU':14}{'Current':8}{'MBU':>6}{'Ach.BW':>8}{'Target':8}{'Savings/mo':>12}{'Savings%':>10}")
        for r in mbu_rightsizing:
            print(f"  {r['gpu_id']:14}{r['current_gpu']:8}{r['mbu']:>6.3f}{r['achieved_bw_tbs']:>7.2f}T{r['rec_gpu']:>8}${r['monthly_saving']:>11,.0f}{r['pct_saving']:>9.1f}%")
        print(f"Total potential monthly right-sizing savings: ${total_monthly_rightsize_savings:,.0f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "vram_pricing": vram_pricing,
        "mbu_rightsizing": mbu_rightsizing,
        "monthly_rightsize_savings": round(total_monthly_rightsize_savings, 2),
    }



if __name__ == "__main__":
    run()
