# Graph scaling reference workloads

No support union, variant grouping, history clone or summary admission algorithm
is changed in T14. Before/after optimization comparison is N/A; these are current
reference costs, not a speedup claim or a universal capacity recommendation.

```sh
go test -run '^$' -bench DeclaredUpperBound -benchmem -benchtime=100ms -count=1 \
  ./graphingest/resolution ./graphingest/resolution/history ./recipe/graphsummary
```

Actual run: darwin/arm64 Apple M1 Max, Go benchmark default GOMAXPROCS 10. Raw
[output](../../docs/task20/acceptance/T14-bench.log) includes ns/op, bytes/op and
allocations. Concurrent verification and short sample time affect elapsed values;
these are reference observations, not statistically isolated comparisons.

| Workload | 64 items ns/op; B/op | 256 items ns/op; B/op |
|---|---|---|
| Full Resolver, same-kind one canonical group, equivalent attributes | 138390; 223362 | 700772; 933053 |
| Full Resolver, same group, all distinct attribute variants | 241468; 255031 | 3011140; 1828470 |
| History isolated stable-order union | 49322; 42915 | 279702; 190386 |
| Summary isolated stable-order union | 27865; 42916 | 231161; 190371 |
| Summary eight refresh passes, no-I/O admission | 245087; 472906 | 995059; 1847138 |

Resolver sets MaxEntities=MaxSupports to the workload size, MaxRelations=1 and
admits exactly that many valid entities/support occurrences. Both paths merge
one canonical entity: equivalent attributes stress support union, distinct
attributes stress repeated host Equivalent/clone comparisons. Attribute clone
owns a small slice; domain validators/source admission are cheap fixture ports.
These admitted counts are exact declared maxima in this benchmark profile.

History/summary union benchmarks isolate their actual private kernel over unique
valid locators. They omit history Marshal/decode/fsync and model dispatch. Summary
refresh kernel performs eight passes (512 or 2048 actual callbacks), retains the
MappedText ownership copying, and uses a no-I/O host admission. It does not measure
a complete Global run, external policy latency, provider tokens or network I/O.
Use full-host profiles before raising real capacities or claiming throughput.

Repeated linear support Contains and same-kind variant searches may require
quadratic comparisons. Complete summary runs additionally refresh the whole input
across calls, O(C×S) host admission work. Future optimizations need larger measured
host profiles and differential tests for stable ordering, duplicates, nil/empty,
attribute equality, ownership, admission freshness and cancellation. Finite caps
bound admitted counts, not byte/CPU/RSS cost or codec work.
