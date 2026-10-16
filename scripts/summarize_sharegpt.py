"""Validate and summarize SGLang ShareGPT records; never rewrite raw results."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def percentile(values, pct):
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(path, expected_requests=1000, expected_concurrency=16):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Expected exactly one benchmark record: %s" % path)
    row = json.loads(lines[0])
    problems = []
    if row.get("dataset_name") != "sharegpt":
        problems.append("wrong dataset")
    if row.get("max_concurrency") != expected_concurrency:
        problems.append("wrong concurrency")
    if row.get("completed") != expected_requests:
        problems.append("incomplete requests")
    for key in ("input_lens", "output_lens", "ttfts", "itls", "errors", "generated_texts"):
        if not isinstance(row.get(key), list) or len(row[key]) != expected_requests:
            problems.append("missing/wrong-sized details: " + key)
    errors = row.get("errors", [])
    if any(errors):
        problems.append("request errors")
    inputs, outputs = row.get("input_lens", []), row.get("output_lens", [])
    lengths_valid = all(isinstance(v, int) and v > 0 for v in inputs + outputs)
    if not lengths_valid:
        problems.append("invalid token lengths")
    if lengths_valid and sum(outputs) != row.get("total_output_tokens"):
        problems.append("output token sum mismatch")
    for key in ("duration", "output_throughput", "mean_ttft_ms", "mean_tpot_ms"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append("invalid metric: " + key)
    duration = row.get("duration")
    throughput = row.get("output_throughput")
    if isinstance(duration, (int, float)) and duration > 0 and isinstance(throughput, (int, float)):
        if not math.isclose(sum(outputs) / duration, throughput, rel_tol=1e-6):
            problems.append("throughput calculation mismatch")
    ttfts = row.get("ttfts", [])
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in ttfts):
        problems.append("invalid request TTFT")
    result = {
        "source": str(path), "valid": not problems, "problems": problems,
        "completed": row.get("completed"), "concurrency": row.get("max_concurrency"),
        "duration_s": duration, "output_tps_total": throughput,
        "output_tps_per_card": throughput / 2 if isinstance(throughput, (int, float)) else None,
        "ttft_mean_ms": row.get("mean_ttft_ms"), "tpot_mean_ms": row.get("mean_tpot_ms"),
        "accept_length": row.get("accept_length"),
        "total_input_tokens": row.get("total_input_tokens"),
        "total_output_tokens": row.get("total_output_tokens"),
        "input_lengths_sha256": hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest(),
    }
    if not problems:
        result["ttft_p95_ms"] = percentile(ttfts, .95) * 1000
        result["mean_input_length"] = sum(inputs) / len(inputs)
        result["mean_output_length"] = sum(outputs) / len(outputs)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        result = summarize(path, args.requests, args.concurrency)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failed = failed or not result["valid"]
    raise SystemExit(1 if failed else 0)
