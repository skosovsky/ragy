# Persistent dense and tensor comparison

Run from `examples/conformance`:

```sh
go run ./tensor_comparison -fixture tensor_comparison/fixture.json -output /tmp/tensor-comparison.json
```

The saved fixture contains normalized float32 dense vectors, token matrices and
ordinal qrels. The executable stages and publishes actual local-filesystem dense
and tensor targets through durable lifecycle manifests, then constructs fresh
adapter instances and pinned scoped read bindings. Namespace/source/revision/access
and model/space identities are explicit in the example and output configuration.
Dense TopK=10 is measured separately from dense candidates=100 → bounded MaxSim
TopK=10. Each sample reports actual native rankings, candidates and total paired
latencies, including candidate retrieval in the tensor path. Query/doc matrices are
supplied by the fixture; no live embedding/model call or tokenizer is involved.

Recall@10 counts grade>0 judgments; nDCG@10 uses exponential gain and full-corpus
ideal ranking. Candidate recall uses the same relevant artifact set. A negative run
removes t1 from the actual dense candidates and demonstrates candidate loss. Raw
native MaxSim scores remain 2, 1 and -1. Embedding sizes count float32 vector payload
bytes; index sizes sum actual logical files in target roots (catalog and payloads).
Manifest storage is separate from those target-file measurements. Runtime model
calls/input/output tokens are zero for this saved-embedding path.

The reference report shows baseline nDCG=0.7098097414 and tensor nDCG=1, with Recall=1
and candidate recall=1. Removing t1 yields Recall/candidate recall=0.5. These synthetic
embeddings intentionally exercise ordering and candidate loss; they prove no
production model quality. A single query with five repeats over three documents is
insufficient for meaningful p50/p95. The report retains raw timings, null percentiles
and an unavailable latency gate. Quality thresholds (+0.05 nDCG and candidate recall
at least 0.95) are computed, but default recommendation remains false because the
p95≤2× baseline gate cannot be established. Baseline defaults remain host-owned.

The program uses an owned temporary directory and removes it after output. The
reported corpus and qrels are public synthetic data. Supported execution platforms
are the actual persistent adapters' local filesystem platforms. This is a bounded
consumer example, with no experiment framework, hidden retry or background worker.

Each fixture document requires an explicit ordinal grade. Missing judgments are
rejected for this fixed-qrels comparison and yield unavailable (null) metrics in
metric checks; they are never silently converted to zero relevance. Empty gold
relevance has no defined Recall/nDCG denominator. The positive saved fixture contains
explicit relevant judgments, so its reported metrics are numerical observations.

The report embeds the complete normalized fixture/qrels digest, actual local adapter
identities, scoring modes, query/candidate/repetition settings, deadline and storage
bounds, byte representation and quality thresholds. Config identity is its SHA-256.
Seed policy explicitly states saved-hand-defined-data-no-random-generator; no random
sampling or model generation occurs. Both actual scope snapshots and full source
references on every returned ranking item are retained alongside publication refs.
The loader rejects oversized inputs even when a valid JSON prefix fits inside the
limit; trailing whitespace also counts toward the declared input byte budget.
