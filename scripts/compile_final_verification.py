"""Combine every requested tier, retaining the C28 stability failure and resume."""
import argparse
import json
from pathlib import Path


TIERS = [
    ("BF16_4K_c16", "bf16", 4096, 16),
    ("INT8_4K_c16", "int8", 4096, 16),
    ("INT8_4K_c28", "int8", 4096, 28),
    ("BF16_64K_c2", "bf16", 65536, 2),
    ("INT8_64K_c2", "int8", 65536, 2),
    ("INT8_C27_c27", "int8", 4096, 27),
]


def assignments(path):
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
                if "=" in line and " " not in line.split("=", 1)[0])


def compile_results(batch):
    original = assignments(batch / "batch.txt")
    resumed = assignments(batch / "resume.txt")
    if resumed.get("status") != "0" or "all_remaining_stages_complete" not in resumed:
        raise ValueError("Resume batch is not complete")
    rows = []
    for name, precision, length, concurrency in TIERS:
        path = batch / f"{name}_requests_audit_v2.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if (not row["request_records_valid"] or row["input_len"] != length
                or row["concurrency"] != concurrency or len(row["rounds"]) != 3):
            raise ValueError(f"Invalid or unexpected tier: {path}")
        row["precision"] = precision
        row["source_audit"] = path.name
        rows.append(row)
    picks = {}
    for precision in ("bf16", "int8"):
        for length in (4096, 65536):
            group = [row for row in rows if row["precision"] == precision and row["input_len"] == length]
            picks[f"{precision}_{length}"] = {}
            for criterion in ("all_three_rounds_sla_pass", "pooled_mean_sla_pass"):
                candidates = [row for row in group if row[criterion]]
                best = max(candidates, key=lambda row: row["output_tps_mean_per_card"]) if candidates else None
                picks[f"{precision}_{length}"][criterion] = (
                    {key: best[key] for key in ("run_id", "concurrency", "output_tps_mean_per_card",
                                                "arithmetic_uplift_vs_official_bf16_percent")}
                    if best else None)
    return {"batch_id": batch.name, "all_requested_tiers_completed_and_audited": True,
            "original_batch_status": int(original["status"]), "resume_status": int(resumed["status"]),
            "interruption_note": "Original batch stopped on a busy-DCU guard after BF16 cleanup; idle confirmed before resuming, with frozen server/benchmark scripts unchanged",
            "sla_note": "All-round pass is a conservative stability check; pooled means are also provided because official cross-round aggregation has not been confirmed",
            "scope_note": "Best throughput among these verified tiers, not a global maximum; other fixed tiers and ShareGPT still have single-round screening data",
            "best_among_verified_tiers": picks, "tiers": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(compile_results(args.batch), ensure_ascii=False, indent=2))
