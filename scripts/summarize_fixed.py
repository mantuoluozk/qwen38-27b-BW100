"""Summarize archived fixed-token results without modifying source records."""
import argparse
import json
from pathlib import Path


def summarize(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    rounds = result.get("rounds", [])
    valid = result.get("status") == "PASS" and bool(rounds)
    valid = valid and len(rounds) == result.get("measure_rounds")
    for row in rounds:
        valid = valid and row.get("status") == "PASS"
        valid = valid and row.get("exact_length") == row.get("requested")
        valid = valid and row["ttft_s"]["count"] == row["requested"]
        valid = valid and row["tpot_ms"]["count"] == row["requested"]
    length = result["input_len"]
    if length == 4096:
        old, new = (3.0, 50.0), (5.0, 50.0)
    elif length == 65536:
        old, new = (15.0, 50.0), (20.0, 60.0)
    else:
        old = new = None

    def sla(limits, field):
        if limits is None or not valid:
            return None
        return all(row["ttft_s"][field] < limits[0]
                   and row["tpot_ms"][field] < limits[1] for row in rounds)

    output = {
        "source": str(path), "valid": bool(valid),
        "input_len": length, "output_len": result["output_len"],
        "concurrency": result["concurrency"],
        "round_count": len(rounds),
        "old_sla_p95_all_rounds": sla(old, "p95"),
        "new_sla_mean_all_rounds": sla(new, "mean"),
        "new_sla_p95_all_rounds": sla(new, "p95"),
        "rounds": [{"output_tps_total": row["output_throughput_tps"],
                    "output_tps_per_card": row["output_throughput_tps"] / 2,
                    "ttft_mean_s": row["ttft_s"]["mean"],
                    "ttft_p95_s": row["ttft_s"]["p95"],
                    "tpot_mean_ms": row["tpot_ms"]["mean"],
                    "tpot_p95_ms": row["tpot_ms"]["p95"],
                    "exact": row["exact_length"], "requests": row["requested"]}
                   for row in rounds],
    }
    if rounds:
        output["output_tps_mean_total"] = sum(
            row["output_throughput_tps"] for row in rounds) / len(rounds)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize(path), ensure_ascii=False, indent=2))
