#!/usr/bin/env python3
"""Summarize raw Go benchmark samples; no synthetic extrapolation or SLO."""
import csv
import re
import statistics
from pathlib import Path

results = Path(__file__).parent / 'results'
rows = []
stages = {}
for stage in ('before', 'after'):
    profiles = {}
    for path in sorted(results.glob(f'serial-{stage}-*.txt')):
        for line in path.read_text().splitlines():
            match = re.match(r'^(Benchmark\S+)-4\s+(\d+)\s+(.*)$', line)
            if not match:
                continue
            metrics = {unit: float(value) for value, unit in re.findall(r'([0-9.]+)\s+(ns/op|B/op|allocs/op|snapshot-bytes)', match[3])}
            profiles.setdefault(match[1], []).append(metrics)
    stages[stage] = profiles
    for name, samples in profiles.items():
        latencies = [s['ns/op'] for s in samples]
        row = dict(stage=stage, profile=name, samples=len(samples), ns_median=statistics.median(latencies), ns_min=min(latencies), ns_max=max(latencies), bytes_median=statistics.median(s['B/op'] for s in samples), allocs_median=statistics.median(s['allocs/op'] for s in samples), snapshot_bytes=statistics.median(s['snapshot-bytes'] for s in samples) if 'snapshot-bytes' in samples[0] else '')
        rows.append(row)
if not rows:
    raise SystemExit('No serial measurements found')
with (results / 'scaling.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ['# TASK18 measured scaling', '', 'Medians of three samples; bracketed values are observed min–max. Wall-time measurements on local APFS, GOMAXPROCS4. See workload.md for corpus, admission and storage limits. Setup is excluded. These values do not establish a latency SLO or tail percentile.', '', '| Profile | Before ms/op [min–max] | After ms/op [min–max] | Before → after B/op | Before → after allocs/op |', '|---|---:|---:|---:|---:|']
def cell(samples):
    vals=[s['ns/op']/1e6 for s in samples]
    return f'{statistics.median(vals):.4g} [{min(vals):.4g}–{max(vals):.4g}]'
for name, before in stages['before'].items():
    after=stages['after'].get(name)
    if not after:
        continue
    b=lambda ss,key:f'{statistics.median(s[key] for s in ss):,.0f}'
    lines.append(f'| {name.removeprefix("BenchmarkTask18")} | {cell(before)} | {cell(after)} | {b(before,"B/op")} → {b(after,"B/op")} | {b(before,"allocs/op")} → {b(after,"allocs/op")} |')
(results / 'scaling.md').write_text('\n'.join(lines)+'\n')

process_rows=[]
for path in sorted(results.glob('serial-*-process-stats.txt')):
    raw=path.read_text()
    cpu=re.search(r'([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys',raw)
    rss=re.search(r'(\d+)\s+maximum resident set size',raw)
    if cpu and rss:
        process_rows.append(dict(profile=path.stem.removesuffix('-process-stats'), wall_seconds=cpu[1], user_cpu_seconds=cpu[2], system_cpu_seconds=cpu[3], max_rss_platform_units=rss[1]))
if process_rows:
    with (results/'process-costs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(process_rows[0]))
        writer.writeheader();writer.writerows(process_rows)
    lines=['# Whole-process resource costs','', 'Recorded by Darwin `/usr/bin/time -l`. CPU seconds and maximum RSS include fixture creation/publication, filesystem synchronization, cleanup and any compiler work. They are process costs, not per-query budgets; raw platform RSS values are retained without cross-platform normalization. Repeated N10k dense staging dominates elapsed process duration but is outside query timers.','', '| Profile | Wall seconds | User CPU seconds | System CPU seconds | Maximum RSS (raw platform units) |','|---|---:|---:|---:|---:|']
    for row in process_rows:
        lines.append('| '+' | '.join(str(row[k]) for k in row)+' |')
    (results/'process-costs.md').write_text('\n'.join(lines)+'\n')

retirement=results/'serial-current-retirement.txt'
if retirement.exists():
    profiles={}
    for line in retirement.read_text().splitlines():
        m=re.match(r'^(BenchmarkTask18Retirement\S+)-4\s+\d+\s+(.*)$',line)
        if m:
            values={unit:float(value) for value,unit in re.findall(r'([0-9.]+)\s+(ns/op|B/op|allocs/op|snapshot-bytes)',m[2])}
            profiles.setdefault(m[1],[]).append(values)
    lines=['# Current-format retirement layout comparison','','Same confirmed-cleaned old/tombstone-owner pair fixture in the current implementation. Before/after here means inventory compaction layout; it is separate from the original active-publication baseline. Maintenance is excluded from measured ordinary CAS. Preserved identities, exact reference fences and cleanup receipts remain in snapshot storage.','','| Profile | Samples | Median ms/op [min–max] | Median B/op | Median allocs/op | Snapshot bytes |','|---|---:|---:|---:|---:|---:|']
    for name,samples in profiles.items():
        vals=[s['ns/op']/1e6 for s in samples]
        med=lambda key:statistics.median(s[key] for s in samples)
        lines.append(f'| {name.removeprefix("BenchmarkTask18Retirement/")} | {len(samples)} | {statistics.median(vals):.4g} [{min(vals):.4g}–{max(vals):.4g}] | {med("B/op"):,.0f} | {med("allocs/op"):,.0f} | {med("snapshot-bytes"):,.0f} |')
    (results/'retirement-layout.md').write_text('\n'.join(lines)+'\n')
