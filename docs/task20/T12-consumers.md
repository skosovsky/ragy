# T12 consumer inventory

Baseline `63e775d8e1faee624d04429a63e6958b1efa30a3`; captured before deletion or migration. Repository/examples inventory excludes historical task20 reports. Unknown external consumers require documented clean break.

Search: `rg -n "ragy/ranking|ranking\.|UnsupportedQueryEncoderBridge|preserveExecutionMeta|syncPartialFailureResult" . --glob "*.go" --glob "*.md" --glob "!docs/task20/**"`.

```text
./retrieval/execution.go:966:	result.Executed = preserveExecutionMeta(exec, result.Executed)
./retrieval/execution.go:993:func preserveExecutionMeta[TExecMeta any](incoming, returned TExecMeta) TExecMeta {
./retrieval/execution.go:1262:				return result, errors.Join(syncPartialFailureResult(retrieveErr, final), postErr)
./retrieval/execution.go:1275:		return result, syncPartialFailureResult(retrieveErr, final)
./retrieval/orchestrator.go:966:				return result, errors.Join(syncPartialFailureResult(partialErr, final), postErr)
./retrieval/orchestrator.go:979:		return result, syncPartialFailureResult(partialErr, final)
./retrieval/partial_failure.go:38:// syncPartialFailureResult updates PartialFailureError.Result to match the post-processed set.
./retrieval/partial_failure.go:39:func syncPartialFailureResult[TMeta any](err error, rs ResultSet[TMeta]) error {
./retrieval/partial_failure_test.go:78:		err := syncPartialFailureResult(ragy.ErrUnavailable, rs)
./retrieval/partial_failure_test.go:98:		updated := syncPartialFailureResult(partial, rs)
./retrieval/partial_failure_test.go:110:		if syncPartialFailureResult(nil, rs) != nil {
./retrieval/partial_failure_test.go:111:			t.Fatal("syncPartialFailureResult(nil) = err, want nil")
./retrieval/partial_failure_test.go:122:		updated := syncPartialFailureResult(partial, empty)
./examples/conformance/graph_comparison/README.md:161:an output order, not a native graph similarity ranking. Baseline dense, lexical and
./examples/conformance/graph_comparison/graph_stages_unix.go:17:// Graph facts are unranked; snapshot iteration order is not a similarity ranking.
./examples/conformance/recipe_comparison/README.md:29:usage or an empty successful ranking.
./docs/task12/verification.md:2565:actual scope snapshots and full exact source references accompany each ranking.
./adapters/observability/otel/otel.go:15:	"github.com/skosovsky/ragy/ranking"
./adapters/observability/otel/otel.go:379:	next   ranking.QueryReranker[TMeta]
./adapters/observability/otel/otel.go:385:	next ranking.QueryReranker[TMeta],
./adapters/observability/otel/otel.go:399:// Rerank implements ranking.QueryReranker.
./adapters/observability/otel/otel.go:405:	ctx, span := w.tracer.Start(ctx, "ragy.ranking.rerank")
./adapters/observability/otel/otel.go:473:	next   ranking.Merger[TMeta]
./adapters/observability/otel/otel.go:478:func WrapMerger[TMeta any](next ranking.Merger[TMeta], tracer trace.Tracer) (*Merger[TMeta], error) {
./adapters/observability/otel/otel.go:490:// Merge implements ranking.Merger.
./adapters/observability/otel/otel.go:495:	ctx, span := w.tracer.Start(ctx, "ragy.ranking.merge")
./adapters/observability/otel/otel.go:511:	_ ranking.QueryReranker[any]       = (*QueryReranker[any])(nil)
./adapters/observability/otel/otel.go:512:	_ ranking.Merger[any]              = (*Merger[any])(nil)
./docs/task12/migration.md:319:Preserve mappings/supports through store projection, grouping and reranking. The actual PDF path tests parser output, typed layout resolution, original/derived rendering and persistent dense lifecycle publication. Reparse old documents with unknown coordinates; no API change certifies historical raw records automatically.
./docs/task12/migration.md:563:error, not a silent sample/truncated ranking.
./adapters/observability/otel/otel_test.go:18:	"github.com/skosovsky/ragy/ranking"
./adapters/observability/otel/otel_test.go:301:	runSpanTest(t, "ragy.ranking.rerank", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
./adapters/observability/otel/otel_test.go:320:	runSpanTest(t, "ragy.ranking.merge", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
./adapters/observability/otel/otel_test.go:882:	_ ranking.QueryReranker[contracttest.StructMeta] = (*captureQueryReranker)(nil)
./adapters/observability/otel/otel_test.go:883:	_ ranking.Merger[contracttest.StructMeta]        = (*captureMerger)(nil)
./docs/task12/audits/correctness-joint-recorder-recheck.md:32:Decoder связывает stage names/status, unique actual candidate IDs, unique dense IDs, one-to-one original source support sets между dense/candidate observations и exact delivered WireHit subset. CandidateIDs enumeration хранится отдельно от dense ranking и MaxSim ranking. Candidate-observations действительно собираются из actual query result: score/rank не присваиваются по позиции в CandidateIDs. Fixture intentionally меняет dense vs MaxSim order; native 2/1/-1 сохраняются. CandidateBudget берётся из actual RerankResult, не реконструируется по длине returned list.
./examples/conformance/tensor_comparison/README.md:20:ideal ranking. Candidate recall uses the same relevant artifact set. A negative run
./README.md:317:- `ranking.QueryReranker` and `ranking.Merger` for post-retrieval ranking
./README.md:456:fusion and reranking. Caller-comparator `Rerank` produces explicit rank-only
./adapters/cohere/rerank/client.go:16:	"github.com/skosovsky/ragy/ranking"
./adapters/cohere/rerank/client.go:188:// Rerank implements ranking.QueryReranker.
./adapters/cohere/rerank/client.go:270:var _ ranking.QueryReranker[any] = (*Client[any])(nil)
./adapters/cohere/README.md:5:`Config.Limits` bounds local work. Zero values select 128 inputs (query plus documents), 1 MiB aggregate UTF-8 text, 2 MiB request, 16 MiB response, and 30 seconds. The token-matrix row limit does not apply to reranking. There is one dispatch and no library retry; standard HTTP clients are cloned with redirects disabled. Custom Doer implementations must honor cancellation and must not hide retries or redirects. Raw bodies, credentials, URLs and transport diagnostics are excluded from returned errors. Parent cancellation and deadlines propagate.
./recipe/encoding.go:10:// UnsupportedQueryEncoderBridge documents the strict integration boundary for
./recipe/encoding.go:13:type UnsupportedQueryEncoderBridge struct{ Embedder dense.Embedder }
./recipe/encoding.go:15:func (UnsupportedQueryEncoderBridge) Admit(context.Context, dense.Request) error {
./recipe/encoding.go:18:func (UnsupportedQueryEncoderBridge) Encode(context.Context, dense.Request, ModelLimits) (dense.Result, Usage, error) {
./recipe/README.md:140:its own encoding. `UnsupportedQueryEncoderBridge` rejects supplied adapters without
./recipe/observation_test.go:118:				f.config.QueryEncoder = recipe.UnsupportedQueryEncoderBridge{}
./recipe/delivery_test.go:96:	f.config.QueryEncoder = recipe.UnsupportedQueryEncoderBridge{}
```

Ranking consumers: cohere/rerank, observability/otel, its tests and README. QueryReranker moves to retrieval beside ResultMerger; span names remain stable observability identifiers. Aliases/constructors provide no independent responsibility. No consumer initializes unused Embedder on UnsupportedQueryEncoderBridge.

Result-only concrete nodes/pipeline/builder are package-private; internal tests and execution/admission adapters are callers. They can delegate to NoExecutionMeta execution nodes instead of independent composition algorithms. Migrate behavioral tests rather than discard coverage. Public RequestExecution* remains the common engine.
