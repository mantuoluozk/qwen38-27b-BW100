"""Audit actual supplement requests and generate separately dated result tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


FIXED_BASE = {
    (4096, 1): (21.195, .807, 46.437),
    (4096, 8): (182.734, 3.816, 40.076),
    (4096, 16): (304.034, 6.801, 46.009),
    (4096, 32): (433.847, 12.748, 61.331),
    (4096, 64): (545.911, 24.588, 93.002),
    (4096, 128): (577.456, 60.410, 135.236),
    (65536, 1): (13.616, 19.266, 54.682),
    (65536, 8): (39.128, 86.563, 120.015),
    (65536, 16): (39.199, 243.047, 119.859),
    (65536, 32): (39.201, 556.433, 119.849),
}
SHARE_BASE = {
    1: (21.454, 200.866, 45.727),
    4: (106.207, 189.951, 35.851),
    16: (307.450, 189.460, 49.725),
    32: (435.996, 199.106, 69.259),
    64: (528.158, 215.717, 103.927),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def audit_fixed(path):
    data = load(path)
    config = load(path.parent / 'config.json')
    n = data['requests_per_round']
    assert data['status'] == 'PASS' and data['successful_rounds'] == data['measure_rounds']
    assert data['output_len'] == 1024 and n == max(8, 2 * data['concurrency'])
    assert config['seed'] == 20260912 and config['warmup_rounds'] == data['warmup_rounds'] == 2
    assert len(data['rounds']) == data['measure_rounds']
    hashes = {}
    for phase in ['warmup', 'round']:
        count = data['warmup_rounds'] if phase == 'warmup' else data['measure_rounds']
        requested = data['concurrency'] if phase == 'warmup' else n
        for index in range(1, count + 1):
            folder = path.parent / f'{phase}-{index:02d}'
            rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
            summary = load(folder / 'summary.json')
            assert len(rows) == requested and len({r['request_index'] for r in rows}) == requested
            assert summary['status'] == 'PASS' and summary['exact_length'] == requested
            for row in rows:
                assert row['error'] is None and row['done'] and row['status_code'] == 200
                assert row['prompt_tokens'] == data['input_len'] and row['completion_tokens'] == 1024
                assert row['ttft_s'] is not None and row['tpot_ms'] is not None
            assert abs(statistics.mean(r['ttft_s'] for r in rows) - summary['ttft_s']['mean']) < 1e-7
            assert abs(statistics.mean(r['tpot_ms'] for r in rows) - summary['tpot_ms']['mean']) < 1e-7
            assert abs(requested * 1024 / summary['wall_time_s'] - summary['output_throughput_tps']) < 1e-7
            hashes[f'{phase}-{index:02d}/requests.jsonl'] = sha(folder / 'requests.jsonl')
    limits = (5, 50) if data['input_len'] == 4096 else (20, 60)
    rounds = data['rounds']
    ttft = statistics.mean(r['ttft_s']['mean'] for r in rounds)
    tpot = statistics.mean(r['tpot_ms']['mean'] for r in rounds)
    passing = sum(r['ttft_s']['mean'] < limits[0] and r['tpot_ms']['mean'] < limits[1] for r in rounds)
    return {
        'run_id': path.parent.parent.name, 'input_len': data['input_len'],
        'concurrency': data['concurrency'], 'rounds': len(rounds),
        'output_tps_total': statistics.mean(r['output_throughput_tps'] for r in rounds),
        'ttft_mean_s': ttft, 'tpot_mean_ms': tpot,
        'sla_pass': ttft < limits[0] and tpot < limits[1],
        'sla_pass_rounds': passing, 'measured_requests': n * len(rounds),
        'round_details': [{'ttft_s': r['ttft_s']['mean'], 'tpot_ms': r['tpot_ms']['mean'],
                           'tps_total': r['output_throughput_tps'], 'requested': r['requested']} for r in rounds],
        'source': f'{path.parent.parent.name}/{path.parent.name}/summary.json',
        'source_sha256': sha(path), 'request_sha256': hashes,
        'request_and_length_audit': 'PASS', 'measurement_date': '2026-10-21',
    }


def compare(actual, baseline):
    return (actual[0] / baseline[0] - 1) * 100, (1 - actual[1] / baseline[1]) * 100, (1 - actual[2] / baseline[2]) * 100


def triple(values):
    return f'{values[0]:.3f} / {values[1]:.4f} / {values[2]:.3f}'


def delta(values):
    return ' / '.join(f'{v:+.2f}%' for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--batch', default='SUPPLEMENT_20261021_01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overall = {'date': '2026-10-21', 'batch': args.batch, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'baseline_reference': 'Official BF16 BW100/SGLang mean table in archived 社区qwen3.8挑战赛.md',
               'baseline_url': 'https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003',
               'baseline_choice_confirmed_by_user_20261021': True,
               'original_submission_unchanged': True, 'precisions': {}}
    for precision in ['bf16', 'int8']:
        old = load(args.repository / f'submissions/20261016/result_{precision}.json')
        run = args.runs / f'{args.batch}_{precision}'
        new = []
        if run.exists():
            for source in sorted(run.glob('fixed-*-1024-c*/summary.json')):
                new.append(audit_fixed(source))
        if new:
            expected_config = {'bf16': '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab',
                               'int8': 'c8c89574834abe1bab2fd76b0ca680399ce0707c20b887d3a4890a04e0101a5e'}[precision]
            assert (run / 'model-config.sha256').read_text().split()[0] == expected_config
            assert 'sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad' in (run / 'image.txt').read_text()
        # Preserve the originally reported low-concurrency rows; only add missing tiers.
        fixed = {(r['input_len'], r['concurrency']): dict(r, measurement_date='original_submission')
                 for r in old['fixed_single_round']}
        for row in new:
            key = row['input_len'], row['concurrency']
            assert key not in fixed, f'Do not overwrite original results: {key}'
            fixed[key] = row
        missing = sorted(set(FIXED_BASE) - set(fixed))
        boundaries = [(65536, 3)] + ([(4096, 18), (4096, 19)] if precision == 'bf16' else [])
        missing_boundaries = [key for key in boundaries if key not in fixed]
        complete = not missing and not missing_boundaries
        if not args.allow_incomplete:
            assert complete, f'{precision}: missing {missing}, boundaries {missing_boundaries}'
            assert (run / 'runner.txt').exists() and 'runner_status=0' in (run / 'runner.txt').read_text()
        text = f'# Qwen3.8-27B BW100 SGLang {precision.upper()}：2026-10-21 补充报告\n\n'
        text += '## 一、环境与范围\n\n'
        text += f'论坛 ID：zack233。BW100 双卡 IFB TP2，DCU2/3，SGLang 0.5.12，DTK26.04。'
        text += f'沿用原 {"BF16 slots32" if precision == "bf16" else "官方 Channel-INT8 W8A8 slots64"} 配置，FA3/HCU布局/连续FA/EAGLE3-1-4，BF16 KV、FP32 SSM、禁用custom all-reduce、原版GEMM。\n\n'
        text += '这是截止后按官方补充要求新增的实测，不冒充10月18日前已有数据。原提交包、原日志及哈希不覆盖；新旧测量日期、轮数、运行ID分别列明。只补固定缺档与SLA边界，不重跑GSM8K或ShareGPT。\n\n'
        text += '## 二、官方基线与计算\n\n'
        text += '基线使用已归档官方BW100/SGLang BF16均值表；所有吞吐为输出tokens/s，主表为双卡总吞吐，除2得单卡。TTFT：固定场景为秒，ShareGPT为毫秒；TPOT均为毫秒。'
        text += '吞吐提升=(实测/基线−1)×100%；时延降低=(1−实测/基线)×100%，负号表示退化。INT8也对同一官方BF16基线，不能称为同INT8权重调参收益。\n\n'
        text += '来源：[官方基线帖](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)及此前归档赛题表。2026-10-21用户确认未收到新版基线，按之前官方均值表计算；不另造基线或擅换分母。\n\n'
        text += '## 三、优化方法与测量\n\n'
        text += '不加入新优化。新增固定缺档2轮预热、1轮正式；SLA边界3轮正式。预热每轮C请求，正式每轮max(8,2C)，随机token-ID精确输入、ignore_eos固定输出1024。客户端读取超时3600秒，不因SLA失败提前结束。每次起服检查所选DCU显存与HCU占用均为0；两精度顺序运行。\n\n'
        text += '## 四、数据与各指标基线对比\n\n'
        text += '### 4.1 官方固定档（基线 / 实测：TPS、TTFT秒、TPOT毫秒）\n\n'
        text += '| 场景/并发 | 官方均值 TPS / TTFT / TPOT | 实测均值 TPS / TTFT / TPOT | TPS提升 / TTFT降低 / TPOT降低 | 均值SLA | 轮数/运行 |\n|---|---|---|---|---|---|\n'
        for key, baseline in FIXED_BASE.items():
            label = f'{"4K" if key[0] == 4096 else "64K"}/1K C{key[1]}'
            if key not in fixed:
                text += f'| {label} | {triple(baseline)} | 待完成 | — | — | — |\n'
                continue
            row = fixed[key]
            actual = row['output_tps_total'], row['ttft_mean_s'], row['tpot_mean_ms']
            row['official_bf16_comparison'] = dict(zip(['tps_gain_percent', 'ttft_reduction_percent', 'tpot_reduction_percent'], compare(actual, baseline)))
            text += f'| {label} | {triple(baseline)} | {triple(actual)} | {delta(compare(actual, baseline))} | {"通过" if row["sla_pass"] else "未通过"} | {row["rounds"]} / {row["run_id"]} |\n'
        text += '\n### 4.2 ShareGPT五档（基线 / 实测：双卡TPS、TTFT毫秒、TPOT毫秒）\n\n'
        text += '| 并发 | 官方均值 TPS / TTFT / TPOT | 实测均值 TPS / TTFT / TPOT | TPS提升 / TTFT降低 / TPOT降低 | 正式请求/轮数 |\n|---|---|---|---|---|\n'
        for row in old['sharegpt_single_round']:
            baseline = SHARE_BASE[row['concurrency']]
            actual = row['output_tps_total'], row['ttft_mean_ms'], row['tpot_mean_ms']
            row['official_bf16_comparison'] = dict(zip(['tps_gain_percent', 'ttft_reduction_percent', 'tpot_reduction_percent'], compare(actual, baseline)))
            text += f'| C{row["concurrency"]} | {triple(baseline)} | {triple(actual)} | {delta(compare(actual, baseline))} | 1000/1000，1轮 |\n'
        text += '\nShareGPT沿用原件，没有重测；公开文件哈希与样本指纹见原包。不套用固定长度SLA。\n\n'
        text += '### 4.3 SLA边界与最大有效吞吐分开\n\n'
        text += '4K mean TTFT<5s且mean TPOT<50ms；64K <20s且<60ms。三轮平均TTFT/TPOT与每轮判定均保留；同请求数下轮均值平均等于请求合并均值。最大通过并发不等于最大通过吞吐。\n\n'
        text += '| 场景/并发 | 双卡 / 单卡TPS | mean TTFT秒 / TPOT毫秒 | 合并均值SLA | 逐轮通过 | 日期/运行 |\n|---|---|---|---|---|---|\n'
        evidence = [dict(r, measurement_date='original_submission') for r in old['fixed_three_round']] + [r for r in new if r['rounds'] == 3]
        for row in sorted(evidence, key=lambda r: (r['input_len'], r['concurrency'])):
            text += f'| {"4K" if row["input_len"] == 4096 else "64K"}/1K C{row["concurrency"]} | {row["output_tps_total"]:.4f} / {row["output_tps_total"]/2:.4f} | {row["ttft_mean_s"]:.4f} / {row["tpot_mean_ms"]:.4f} | {"通过" if row["sla_pass"] else "未通过"} | {row["sla_pass_rounds"]}/{row["rounds"]} | {row["measurement_date"]} / {row["run_id"]} |\n'
        text += '\n满足SLA的已测最大并发汇总（以三轮证据为主，基线由官方固定表均值重算）：\n\n'
        text += '| 场景 | 基线并发 | 本方案合并均值最大已测并发 | 该档TTFT秒 / TPOT毫秒 | 并发提升 | 三轮全部通过的最大已测并发 | SLA内最高三轮平均单卡吞吐 / 提升 |\n|---|---:|---:|---|---:|---:|---|\n'
        maximum_summary = []
        for length in [4096, 65536]:
            passing = [r for r in evidence if r['input_len'] == length and r['sla_pass']]
            stable = [r for r in passing if r['sla_pass_rounds'] == r['rounds']]
            if not passing:
                continue
            highest = max(passing, key=lambda r: r['concurrency'])
            peak = max(passing, key=lambda r: r['output_tps_total'])
            strict_max = max((r['concurrency'] for r in stable), default=0)
            base_c = 8 if length == 4096 else 1
            base_tps = FIXED_BASE[(length, base_c)][0] / 2
            maximum_summary.append({'input_len': length, 'baseline_concurrency_recomputed': base_c,
                                    'max_measured_passing_concurrency': highest['concurrency'],
                                    'max_measured_all_rounds_pass_concurrency': strict_max,
                                    'max_passing_mean_output_tps_per_card': peak['output_tps_total'] / 2,
                                    'throughput_gain_percent': (peak['output_tps_total'] / 2 / base_tps - 1) * 100,
                                    'ttft_s_at_max_concurrency': highest['ttft_mean_s'],
                                    'tpot_ms_at_max_concurrency': highest['tpot_mean_ms']})
            text += f'| {"4K" if length == 4096 else "64K"}/1K | {base_c} | {highest["concurrency"]} | {highest["ttft_mean_s"]:.4f} / {highest["tpot_mean_ms"]:.4f} | {(highest["concurrency"]/base_c-1)*100:+.2f}% | {strict_max} | {peak["output_tps_total"]/2:.4f} / {(peak["output_tps_total"]/2/base_tps-1)*100:+.2f}% |\n'
        text += '\n原赛题另表4K基线并发写C1，与固定表C8的均值通过事实有冲突；本表明确按固定表均值重算。TTFT/TPOT与基线各指标对照见4.1，不把不同输入长度或并发的时延百分比冒称同档提升。最大并发仅限有三轮证据的已测集合，邻近失败/单轮诊断另保留。\n'
        diagnostics = []
        neighbor_run, neighbor_c = ('F027', 20) if precision == 'bf16' else ('I011', 29)
        neighbor = args.runs / neighbor_run / f'fixed-4096-1024-c{neighbor_c}/summary.json'
        if neighbor.exists():
            row = audit_fixed(neighbor)
            row['measurement_date'] = '2026-10-15'
            diagnostics.append(row)
            text += f'\n相邻失败档历史原件：{neighbor_run}，4K C{neighbor_c}，1轮，TTFT {row["ttft_mean_s"]:.4f}s、TPOT {row["tpot_mean_ms"]:.4f}ms，均值SLA未通过。完整原始运行随历史边界证据包附上，不拿单轮失败冒称三轮失败。64K C4历史失败原件见原提交包（BF16 F029 / INT8 I014）。\n'
        if precision == 'bf16':
            text += '\nBF16 C18三轮均过，C19仅1/3轮通过但三轮合并均值通过；C19吞吐较高，但余量小且跨轮波动，不把它称为三轮稳定通过。保持原C16稳妥吞吐参考，另如实展示C18最大三轮全过并发与C19均值边界。\n\n'
        else:
            text += '\nINT8 C28三轮仅2/3通过但合并均值通过，原稳妥4K主报C27（三轮全过）；不隐藏超限轮次。新增64K C3三轮均过，更新该场景最高已测通过档；没有重写原C2结果。\n\n'
        text += '### 4.4 GSM8K与硬件身份\n\n'
        accuracy = old['gsm8k']
        text += f'沿用原最终候选GSM8K：{"1291/1319 = 97.8772%" if precision == "bf16" else "1287/1319 = 97.5739%"}，固定OpenCompass 4-shot CoT、1319题正常结束且无截断。官方补充通知已明确只测试GSM8K，不再要求其他三个数据集。hy-smi原始hardware-declare.txt显示BW100，与申报一致。\n\n'
        text += '## 五、原始证据与边界\n\n'
        text += f'新增原始目录 `{args.batch}_{precision}/` 含服务启动、模型配置哈希、脚本快照/哈希、选卡快照、逐轮请求及汇总、结束时间与状态。结构化结果附所有新请求文件SHA256和审计结论。原报告与完整包仍在 `submissions/20261016/`，新补充原件另行打包，不修改旧附件。\n\n'
        text += f'当前生成状态：{"缺档和计划边界均已完成" if complete else "中间进度，尚有未完成项，不作最终交付"}。'
        text += '部分固定档及ShareGPT仍为单轮；新增边界为三轮，不冒称全档三轮。峰值吞吐不等于SLA有效成绩。Demo未录制视频，本次不重跑精度。\n'
        write_json(args.output / f'result_{precision}.json', {
            'precision': precision, 'official_fixed_complete': not missing,
            'missing_fixed_tiers': missing, 'missing_boundary_tiers': missing_boundaries,
            'fixed': list(fixed.values()), 'boundary_three_round': evidence,
            'sla_maximum_measured': maximum_summary,
            'historical_neighbor_diagnostics': diagnostics,
            'sharegpt': old['sharegpt_single_round'], 'gsm8k': accuracy,
        })
        (args.output / f'{precision}.md').write_text(text, encoding='utf-8')
        overall['precisions'][precision] = {'complete': complete, 'missing_fixed': missing,
                                           'missing_boundary': missing_boundaries,
                                           'new_tiers_audited': len(new),
                                           'new_formal_requests': sum(r['measured_requests'] for r in new),
                                           'historical_neighbor_audited': len(diagnostics)}
    write_json(args.output / 'audit.json', overall)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
