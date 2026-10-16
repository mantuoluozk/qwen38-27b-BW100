"""Audit three-round final runs against their request records; no inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(run_dir, length, concurrency):
    scenario = run_dir / f"fixed-{length}-1024-c{concurrency}"
    summary = read_json(scenario / "summary.json")
    config = read_json(scenario / "config.json")
    expected = max(8, 2 * concurrency)
    problems, rows, artifacts = [], [], {}
    if not (summary["status"] == "PASS" and summary["measure_rounds"] == 3
            and summary["warmup_rounds"] == 2 and summary["successful_rounds"] == 3
            and summary["requests_per_round"] == expected
            and len(summary["rounds"]) == 3):
        problems.append("Wrong round count, request budget or benchmark status")
    if not (config["input_len"] == length and config["output_len"] == 1024
            and config["concurrency"] == concurrency and config["seed"] == 20260912):
        problems.append("Unexpected benchmark configuration")
    limits = (5.0, 50.0) if length == 4096 else (20.0, 60.0)
    for phase, count, requests in (("warmup", 2, concurrency), ("round", 3, expected)):
        for number in range(1, count + 1):
            request_path = scenario / f"{phase}-{number:02d}" / "requests.jsonl"
            details = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines() if line]
            result = read_json(request_path.with_name("summary.json"))
            artifacts[str(request_path.relative_to(run_dir))] = hashlib.sha256(request_path.read_bytes()).hexdigest()
            valid = (len(details) == requests and result["status"] == "PASS"
                     and result["requested"] == requests and result["exact_length"] == requests
                     and result["successful"] == requests
                     and {item["request_index"] for item in details} == set(range(requests)))
            valid = valid and all(item["status_code"] == 200 and item["done"]
                                  and item["error"] is None and item["prompt_tokens"] == length
                                  and item["completion_tokens"] == 1024
                                  and item["ttft_s"] is not None and item["tpot_ms"] is not None
                                  for item in details)
            if not valid:
                problems.append(f"Invalid request records in {phase}-{number:02d}")
                continue
            ttft = sum(item["ttft_s"] for item in details) / requests
            tpot = sum(item["tpot_ms"] for item in details) / requests
            tps = requests * 1024 / result["wall_time_s"]
            if not all(math.isclose(value, reported, rel_tol=1e-8, abs_tol=1e-8) for value, reported in (
                (ttft, result["ttft_s"]["mean"]), (tpot, result["tpot_ms"]["mean"]),
                (tps, result["output_throughput_tps"]))):
                problems.append(f"Request/summary mismatch in {phase}-{number:02d}")
            if phase == "round":
                if result != summary["rounds"][number - 1]:
                    problems.append(f"Per-round/aggregate mismatch in round-{number:02d}")
                rows.append({"round": number, "requests": requests,
                             "output_tps_total": tps, "output_tps_per_card": tps / 2,
                             "ttft_mean_s": ttft, "tpot_mean_ms": tpot,
                             "ttft_p95_s": result["ttft_s"]["p95"],
                             "tpot_p95_ms": result["tpot_ms"]["p95"],
                             "sla_pass": ttft < limits[0] and tpot < limits[1]})
    runner = (run_dir / "runner.txt").read_text(encoding="utf-8")
    if "runner_status=0\n" not in runner:
        problems.append("Runner has not finished successfully")
    mean_total = sum(row["output_tps_total"] for row in rows) / len(rows) if rows else None
    if mean_total is not None and not math.isclose(mean_total, summary["output_throughput_tps"]["mean"], rel_tol=1e-8):
        problems.append("Aggregate throughput mismatch")
    baseline_per_card = 91.367 if length == 4096 else 6.808
    ttft_mean = sum(row["ttft_mean_s"] for row in rows) / len(rows) if rows else None
    tpot_mean = sum(row["tpot_mean_ms"] for row in rows) / len(rows) if rows else None
    return {"run_id": run_dir.name, "input_len": length, "output_len": 1024,
            "concurrency": concurrency, "request_records_valid": not problems,
            "problems": problems, "sla_ttft_limit_s": limits[0], "sla_tpot_limit_ms": limits[1],
            "all_three_rounds_sla_pass": not problems and len(rows) == 3 and all(row["sla_pass"] for row in rows),
            "sla_pass_round_count": sum(row["sla_pass"] for row in rows),
            "mean_of_round_ttft_mean_s": ttft_mean,
            "mean_of_round_tpot_mean_ms": tpot_mean,
            "pooled_mean_sla_pass": not problems and len(rows) == 3 and ttft_mean < limits[0] and tpot_mean < limits[1],
            "sla_aggregation_note": "Equal measured request counts: pooled request means equal the mean of round means; all-round pass is a stricter stability check, not an asserted official scoring rule",
            "output_tps_mean_total": mean_total,
            "output_tps_mean_per_card": mean_total / 2 if mean_total is not None else None,
            "output_tps_per_card_range": [min(row["output_tps_per_card"] for row in rows), max(row["output_tps_per_card"] for row in rows)] if rows else None,
            "official_bf16_baseline_per_card_tps": baseline_per_card,
            "arithmetic_uplift_vs_official_bf16_percent": (mean_total / 2 / baseline_per_card - 1) * 100 if mean_total is not None else None,
            "baseline_note": "4K uses fixed-tier mean-SLA C8 baseline; 64K uses C1; arithmetic comparison, not an official score",
            "rounds": rows, "request_artifact_sha256": artifacts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input-len", type=int, choices=(4096, 65536), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.input_len, args.concurrency)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["request_records_valid"] else 1)
