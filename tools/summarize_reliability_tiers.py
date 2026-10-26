"""Summarize all tasks, assigning zero QoS to reliability-unsatisfied tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from compare_tore import summarize_reliability_tiers, summarize_task_groups, reliability_adjusted_qos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--all-tasks', action='store_true', help='Include all tasks (default)')
    scope.add_argument('--only-satisfied', action='store_true', help='Explicitly restrict performance metrics to satisfied tasks')
    args = parser.parse_args()
    source = args.input.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = pd.read_csv(source, float_precision='round_trip',
                          dtype={'safe_mask':str, 'action_mask':str})
    only_satisfied = args.only_satisfied
    episode, seeds, summary = summarize_reliability_tiers(records, only_satisfied=only_satisfied)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (('tier_episode_metrics', episode), ('tier_seed_summary', seeds),
                        ('tier_summary', summary),
                        ('eval_episode_metrics', summarize_task_groups(records, ['seed_id','episode','method'], only_satisfied=only_satisfied)),
                        ('eval_seed_summary', summarize_task_groups(records, ['seed_id','method'], only_satisfied=only_satisfied)),
                        ('summary', summarize_task_groups(records, ['method'], only_satisfied=only_satisfied))):
        frame.to_csv(out/f'{name}.csv', index=False, float_format='%.17g')
    task_metrics = records[['seed_id','episode','task_id','method','reliability_requirement',
                            'requirement_satisfied','pair_reliability','task_qos','task_latency','is_fallback']].copy()
    task_metrics['reported_qos'] = reliability_adjusted_qos(records)
    task_metrics.to_csv(out/'task_qos_metrics.csv.gz', index=False, float_format='%.17g')
    paired = []
    for requirement, group in seeds.groupby('reliability_requirement'):
        ppo = group[group.method=='PPO'].set_index('seed_id')
        for method in sorted(set(group.method)-{'PPO'}):
            other = group[group.method==method].set_index('seed_id').loc[ppo.index]
            for metric in ('mean_qos', 'mean_latency_s', 'mean_satisfied_reliability'):
                delta = (ppo[metric]-other[metric]).dropna()
                half = (student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
                        if len(delta)>1 else np.nan)
                paired.append(dict(reliability_requirement=requirement, reference=method, metric=metric,
                    delta_ppo_minus_reference=delta.mean(), ci95_lower=delta.mean()-half,
                    ci95_upper=delta.mean()+half, seed_count=len(delta),
                    ppo_better_seeds=int((delta<0 if metric=='mean_latency_s' else delta>0).sum())))
    pd.DataFrame(paired).to_csv(out/'paired_seed_statistics.csv', index=False, float_format='%.17g')
    selected = records.loc[records.requirement_satisfied] if only_satisfied else records
    adjusted = selected.assign(reported_qos=reliability_adjusted_qos(selected))
    direct = adjusted.groupby(['reliability_requirement','method']).reported_qos.agg(['size','mean'])
    indexed = summary.set_index(['reliability_requirement','method']).loc[direct.index]
    np.testing.assert_array_equal(indexed.qos_task_count, direct['size'])
    np.testing.assert_allclose(indexed.mean_qos, direct['mean'], rtol=0, atol=1e-14)
    latency = selected.groupby(['reliability_requirement','method']).task_latency
    np.testing.assert_allclose(indexed.mean_latency_s, latency.mean(), rtol=0, atol=1e-14)
    np.testing.assert_allclose(indexed.p95_latency_s, latency.quantile(.95), rtol=0, atol=1e-14)
    assert summary.qos_task_count.sum()==len(selected)
    empty = summary.qos_task_count.eq(0)
    assert summary.loc[empty, ['mean_qos','mean_latency_s','p95_latency_s']].isna().all().all()
    assert summary.tier_task_count.sum()==len(records)
    assert task_metrics.loc[~task_metrics.requirement_satisfied, 'reported_qos'].eq(0).all()
    np.testing.assert_array_equal(task_metrics.loc[task_metrics.requirement_satisfied, 'reported_qos'],
                                  records.loc[records.requirement_satisfied, 'task_qos'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    metadata = dict(source=str(source), source_sha256=digest,
        grouping='Task reliability_requirement',
        qos_scope='Only requirement_satisfied tasks per tier' if only_satisfied else 'All tasks per tier',
        qos_rule='QoS=raw task_qos when requirement_satisfied; otherwise 0',
        raw_task_qos_preserved=True,
        mean_satisfied_reliability_scope='Selected unified-model reliability among satisfied tasks; empty groups NaN',
        latency_scope='Same selected tasks as QoS', empty_groups='NaN; zero selected count',
        reliable_fallback_included=True, all_fallback_included=not only_satisfied,
        training_or_simulation_rerun=False,
        seed_ids=sorted(records.seed_id.unique().tolist()), task_records=len(records))
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# 四方法按可靠性要求分档统计', '',
        ('按任务要求的可靠性等级分组，每档仅统计统一相关模型下达标的任务，包括达标的 fallback。'
         if only_satisfied else '按任务要求的可靠性等级分组，每档统计全部任务，包括不达标任务和 fallback；可靠性不达标任务的 QoS 记为 0，达标任务保留原始 QoS。'),
        '复用已有执行记录，仅重新统计。QoS、平均时延和 P95 使用同一统计集合；无统计样本记为“无样本”。可靠性达标率的分母为全部任务。', '',
        '|可靠性要求|方法|统计任务数|该档全部任务数|平均QoS|平均时延/s|P95时延/s|可靠性达标率|达标任务平均可靠性|',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        metrics = '|'.join('无样本' if pd.isna(value) else f'{value:.6f}'
                           for value in (row.mean_qos,row.mean_latency_s,row.p95_latency_s))
        lines.append(f'|{row.reliability_requirement:g}|{row.method}|{row.qos_task_count}|'
            f'{row.tier_task_count}|{metrics}|{row.theoretical_rsr:.2%}|'
            f'{"无样本" if pd.isna(row.mean_satisfied_reliability) else f"{row.mean_satisfied_reliability:.8f}"}|')
    lines += ['', f'原始任务记录：`{source}`', '',
              'task_qos_metrics.csv.gz 保留原始 task_qos，并以 reported_qos 记录可靠性不达标置零后的评分；task_latency 保留实际时延。',
              '逐种子结果见 tier_seed_summary.csv；PPO 与各基线的分档配对差及未校正 95% CI 见 paired_seed_statistics.csv。', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'validation.json').write_text(json.dumps(dict(passed=True,
        every_selected_task_included_once=True, zero_sample_groups_preserved=True,
        source_unchanged=True, grouped_means_verified=True,
        latency_means_and_p95_verified=True, unsatisfied_qos_zero=True,
        satisfied_qos_unchanged=True), indent=2)+'\n')
    (out/'output_manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file() and p.name!='output_manifest.json'}, indent=2)+'\n')
    print(summary.to_string(index=False))
    print(f'Complete: {out}')


if __name__=='__main__':
    main()
