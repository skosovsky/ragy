# Long-page mapping validation

T15 changes only validation reuse: OriginalTexts validates retained UTF-8 once per
batch, and MappedText.Validate validates rendered UTF-8 once per snapshot rather
than once per fragment. Every locator/span boundary, exact/derived distinction,
support shape, encounter order and owned output remains validated. layout region
resolution uses the batch. Standalone OriginalText remains independently validating.

Actual before/after run on darwin/arm64 Apple M1 Max, GOMAXPROCS 10:

| Workload | Before ns/op; B/op | After ns/op; B/op |
|---|---|---|
| 512 words / 12800 retained bytes | 26962969; 5110926 | 9373958; 5132101 |
| 2048 words / 51200 retained bytes | 429788500; 21184248 | 13083552; 21266343 |

[Before log](../docs/task20/acceptance/T15-range-before.log),
[after log](../docs/task20/acceptance/T15-range-after.log). Short 100ms samples
provide reference observations, not isolated statistical throughput guarantees.
The larger workload improves measured elapsed time substantially; allocations
remain dominated by immutable mapping/support construction. No memory reduction
or end-to-end parser/OCR/network speedup is claimed.

Before used the then-current per-word OriginalText plus JoinMapped and final Validate.
After uses OriginalTexts plus the same join and final validation. Both construct
all exact spans from repeated multibyte “длинноеслово ” text and verify the joined
text; counts, callback absence and output remain identical. Before source is the
T14 baseline `e5e9d1e18407b1c02a5a4df6be4e3df5cb1f23e5`; the baseline benchmark
source is retained in [task evidence](../docs/task20/T15-before-benchmark.go.txt).

```sh
go test -run '^$' -bench BenchmarkLongPageWordMappings -benchmem -benchtime=100ms -count=1 ./source
```

[Boundary differential tests](original_batch_test.go) compare every valid/invalid
byte interval through single and batch APIs over multibyte/repeated text, including
late error zero payload, nil/empty inventory, encounter order and detached supports.
Existing mapped joins/slices and actual retained layout region tests still run
under race. Host bounds remain needed for bytes, fragments, supports and callback
work: shape validation and support unions are not a universal linear CPU/RSS bound.
