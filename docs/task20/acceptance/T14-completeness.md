# T14 — независимая приёмка полноты

Baseline HEAD `45ef3d6ded230578e8f7572c91ad89c1c6dd39fc`. Приёмщик не участвовал в реализации, исходники не менял, отчёт корректности не читал. Проверены предоставленные AGENTS.md, master task20, исходный graph report, backlog и текущий diff.

**PASS. Полнота критериев: 5/5 × 100 = 100%. Покрытие назначенных источников: 29/29 × 100 = 100%. Не выполнено: 0; заблокировано: 0; SKIP: 0.**

| Критерий | Статус | Проверяемые доказательства |
|---|---|---|
| T14.C01 | выполнено | Все 29 исходных решений имеют явные retain/change/contract, обоснование и существующие evidence; obsolete resolution roadmap заменён actual package links. |
| T14.C02 | выполнено | Cross-document Billing/LedgerDB требует фактического endpoint support источника A; Name canonical vs mention label, entity/relation literals, stable/reflexive generic kinds; history cardinalities/duplicate occurrence budgets, post-Marshal byte cap, darwin/linux и unknown append описаны и сверены с кодом. |
| T14.C03 | выполнено | Summary repeated freshness O(C×S), selected-only future Resolve vs complete model ancestry, owned delivered text, set membership и first MissingCoverage остановка зафиксированы. Новый AAA regression независимо PASS. |
| T14.C04 | выполнено | Fail-zero MaxNodes/MaxEdges, before-dedup MaxAdmissionRecords, seed-only Insufficient и volatile unavailable без fallback сохранены. Build normalized-view validation возвращает original typed Meta; Stage нормализует labels/encoded attrs и отдельно clones Meta. |
| T14.C05 | выполнено | Complete five-type actual managed/lifecycle composition независимо выполнена: published и interrupted-stage PASS. Все восемь scoped packages race PASS. Фактические upper-bound full resolver и isolated union/refresh benchmarks независимо выполнены; алгоритмы не оптимизируются, before/after N/A без speedup claim. |

Проверка C05 не подменяет будущий T22 all-module/live gate. Эти изменения не оптимизируют union/grouping/admission/clone алгоритмы; поэтому before/after сравнение именно оптимизации неприменимо. Текущие benchmark costs имеют явно ограниченные workload и исключённые I/O/serialization/model расходы.

| Исходный пункт | Статус | Disposition и проверенное evidence |
|---|---|---|
| D23 | выполнено | retain: Keep independent source-bound retrieval facts/provenance ports; general orchestration and semantic policies stay host-owned. Evidence: `graphingest/README.md`, `graphingest/composition.md` |
| D24 | выполнено | contract: Keep source-local endpoint closure and existing JSON names; Name is canonical and Unresolved.Kind has entity/relation literals. Cross-document example explains real evidence requirement. Evidence: `graphingest/materialization/README.md`, `graphingest/resolution/README.md`, `graphingest/materialization/materialization_test.go` |
| D25 | выполнено | contract: Stable reflexive comparable kinds and faithful bounded codecs are host requirements; actual resolver records are declared decisions, with explicit multidimensional occurrence accounting. Evidence: `graphingest/resolution/README.md`, `graphingest/resolution/history/README.md`, `graphingest/resolution/history/history.go` |
| D26 | выполнено | contract: Accepted serialized-byte cap is post-Marshal, not peak-memory; portable snapshots versus darwin/linux local FileStore, durable root and unknown append inspection explicit. Evidence: `graphingest/resolution/history/README.md`, `graphingest/resolution/history/filestore.go` |
| D27 | выполнено | retain: No speculative indexing/copy optimization. Actual declared-max resolver and isolated union/admission benchmarks recorded; stable ownership/order/freshness retained, before/after optimization N/A. Evidence: `graphingest/resolution/scaling.md`, `docs/task20/acceptance/T14-bench.log` |
| D29 | выполнено | contract: Retain selected-citation future Resolve policy. Unselected inputs are not complete derivation dependencies; full revocation requires externally retained map/reduction ancestry, with boundary regression. Evidence: `recipe/graphsummary/README.md`, `recipe/graphsummary/citation_policy_test.go` |
| D30 | выполнено | contract: Fresh Resolve checks original binding and selected supports; delivered data is owned. Membership is declared set coverage; first incomplete community stops later maps/reduce without repair. Evidence: `recipe/graphsummary/README.md`, `recipe/graphsummary/negative_test.go`, `recipe/graphsummary/citation_policy_test.go` |
| D31 | выполнено | retain: Keep seed-only Insufficient recipe heuristic, fail-zero capacities and volatile unavailable without fallback; explicit smaller depth versus capacity guidance. Evidence: `recipe/graphexpand/README.md`, `graph/managed/README.md`, `recipe/graphexpand/expand_test.go` |
| D33 | выполнено | contract: Build returns original owned projected metadata after normalized-view validation; managed Stage sorts/compacts labels and captures normalized filter attributes, retaining separately cloned typed metadata. Full existing deterministic five-type example remains runnable with all callback and lifecycle handoff guidance. Evidence: `graphingest/materialization/README.md`, `graphingest/composition.md`, `graphingest/pipeline_integration_test.go` |
| graph:01 | выполнено | retain: Keep independent source-bound retrieval facts/provenance ports; general orchestration and semantic policies stay host-owned. Evidence: `graphingest/README.md`, `graphingest/composition.md` |
| graph:02 | выполнено | change: Replace obsolete resolution roadmap with actual current extraction/materialization/history/summary package boundaries and runnable handoff link. Evidence: `graphingest/resolution/README.md`, `graphingest/composition.md` |
| graph:03 | выполнено | contract: Keep source-local endpoint closure and existing JSON names; Name is canonical and Unresolved.Kind has entity/relation literals. Cross-document example explains real evidence requirement. Evidence: `graphingest/materialization/README.md`, `graphingest/resolution/README.md`, `graphingest/materialization/materialization_test.go` |
| graph:04 | выполнено | contract: Keep source-local endpoint closure and existing JSON names; Name is canonical and Unresolved.Kind has entity/relation literals. Cross-document example explains real evidence requirement. Evidence: `graphingest/materialization/README.md`, `graphingest/resolution/README.md`, `graphingest/materialization/materialization_test.go` |
| graph:05 | выполнено | contract: Stable reflexive comparable kinds and faithful bounded codecs are host requirements; actual resolver records are declared decisions, with explicit multidimensional occurrence accounting. Evidence: `graphingest/resolution/README.md`, `graphingest/resolution/history/README.md`, `graphingest/resolution/history/history.go` |
| graph:06 | выполнено | contract: Keep source-local endpoint closure and existing JSON names; Name is canonical and Unresolved.Kind has entity/relation literals. Cross-document example explains real evidence requirement. Evidence: `graphingest/materialization/README.md`, `graphingest/resolution/README.md`, `graphingest/materialization/materialization_test.go` |
| graph:07 | выполнено | contract: Stable reflexive comparable kinds and faithful bounded codecs are host requirements; actual resolver records are declared decisions, with explicit multidimensional occurrence accounting. Evidence: `graphingest/resolution/README.md`, `graphingest/resolution/history/README.md`, `graphingest/resolution/history/history.go` |
| graph:08 | выполнено | contract: Stable reflexive comparable kinds and faithful bounded codecs are host requirements; actual resolver records are declared decisions, with explicit multidimensional occurrence accounting. Evidence: `graphingest/resolution/README.md`, `graphingest/resolution/history/README.md`, `graphingest/resolution/history/history.go` |
| graph:09 | выполнено | contract: Accepted serialized-byte cap is post-Marshal, not peak-memory; portable snapshots versus darwin/linux local FileStore, durable root and unknown append inspection explicit. Evidence: `graphingest/resolution/history/README.md`, `graphingest/resolution/history/filestore.go` |
| graph:10 | выполнено | contract: Accepted serialized-byte cap is post-Marshal, not peak-memory; portable snapshots versus darwin/linux local FileStore, durable root and unknown append inspection explicit. Evidence: `graphingest/resolution/history/README.md`, `graphingest/resolution/history/filestore.go` |
| graph:11 | выполнено | retain: No speculative indexing/copy optimization. Actual declared-max resolver and isolated union/admission benchmarks recorded; stable ownership/order/freshness retained, before/after optimization N/A. Evidence: `graphingest/resolution/scaling.md`, `docs/task20/acceptance/T14-bench.log` |
| graph:13 | выполнено | retain: No speculative indexing/copy optimization. Actual declared-max resolver and isolated union/admission benchmarks recorded; stable ownership/order/freshness retained, before/after optimization N/A. Evidence: `graphingest/resolution/scaling.md`, `docs/task20/acceptance/T14-bench.log` |
| graph:14 | выполнено | contract: Retain selected-citation future Resolve policy. Unselected inputs are not complete derivation dependencies; full revocation requires externally retained map/reduction ancestry, with boundary regression. Evidence: `recipe/graphsummary/README.md`, `recipe/graphsummary/citation_policy_test.go` |
| graph:15 | выполнено | contract: Fresh Resolve checks original binding and selected supports; delivered data is owned. Membership is declared set coverage; first incomplete community stops later maps/reduce without repair. Evidence: `recipe/graphsummary/README.md`, `recipe/graphsummary/negative_test.go`, `recipe/graphsummary/citation_policy_test.go` |
| graph:16 | выполнено | contract: Fresh Resolve checks original binding and selected supports; delivered data is owned. Membership is declared set coverage; first incomplete community stops later maps/reduce without repair. Evidence: `recipe/graphsummary/README.md`, `recipe/graphsummary/negative_test.go`, `recipe/graphsummary/citation_policy_test.go` |
| graph:17 | выполнено | retain: Keep seed-only Insufficient recipe heuristic, fail-zero capacities and volatile unavailable without fallback; explicit smaller depth versus capacity guidance. Evidence: `recipe/graphexpand/README.md`, `graph/managed/README.md`, `recipe/graphexpand/expand_test.go` |
| graph:18 | выполнено | retain: Keep seed-only Insufficient recipe heuristic, fail-zero capacities and volatile unavailable without fallback; explicit smaller depth versus capacity guidance. Evidence: `recipe/graphexpand/README.md`, `graph/managed/README.md`, `recipe/graphexpand/expand_test.go` |
| graph:19 | выполнено | retain: Keep seed-only Insufficient recipe heuristic, fail-zero capacities and volatile unavailable without fallback; explicit smaller depth versus capacity guidance. Evidence: `recipe/graphexpand/README.md`, `graph/managed/README.md`, `recipe/graphexpand/expand_test.go` |
| graph:21 | выполнено | contract: Build returns original owned projected metadata after normalized-view validation; managed Stage sorts/compacts labels and captures normalized filter attributes, retaining separately cloned typed metadata. Full existing deterministic five-type example remains runnable with all callback and lifecycle handoff guidance. Evidence: `graphingest/materialization/README.md`, `graphingest/composition.md`, `graphingest/pipeline_integration_test.go` |
| graph:22 | выполнено | contract: Build returns original owned projected metadata after normalized-view validation; managed Stage sorts/compacts labels and captures normalized filter attributes, retaining separately cloned typed metadata. Full existing deterministic five-type example remains runnable with all callback and lifecycle handoff guidance. Evidence: `graphingest/materialization/README.md`, `graphingest/composition.md`, `graphingest/pipeline_integration_test.go` |

Graph:12/D28 (full extraction input byte semantics) и graph:20/D32 (error precedence) назначены другим задачам; не включены в 29 T14 источников.

Независимые проверки, exit 0:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./graphingest/... ./graph/managed ./recipe/graphsummary ./recipe/graphexpand`: `T14-completeness-race.log`, все 8 packages PASS.
- Fresh race verbose actual handoff и citation policy: `T14-completeness-focused.log`, обе handoff ветви и selected/unselected regression PASS.
- `go test -run ^$ -bench DeclaredUpperBound -benchmem -benchtime=100ms -count=1` для resolver/history/summary: `T14-completeness-bench.log`, все benchmark workloads PASS. Проверены размер 64/256, MaxEntities=MaxSupports exact workload maxima, all-distinct variants и same variants, actual private union kernels, восемь refresh passes с exact size×8 admission calls.
- `git diff --check`: PASS.

Свежие runnable tests используют реальные локальные managed/filestore/lifecycle ports и deterministic model fixture. Они не доказывают production remote/provider или hardware power-loss; guide явно это ограничивает.

SHA256 текущих changed/new implementation/test/contract/currentguide файлов (mutable journal/backlog/traceability/acceptance исключены):

| Файл | SHA256 |
|---|---|
| `docs/contracts/remediation.md` | `e942cf94ae418240a4f1377411463f21a8a42601cbaec0f38224cb79b6487cf9` |
| `graphingest/README.md` | `285f63ee422d28b32fb6f97b8c53448816a8ab4476f0be47638409d83ca8e881` |
| `graphingest/composition.md` | `ff1ef6e7cb45f14c40eb1a507ac2e6eac03fd4f2e1a23b8edb43f09d8046bc28` |
| `graphingest/materialization/README.md` | `a57fd18f4bdb35e15e72fd2d224ebc55181627def40e870d60da470b28cf53cb` |
| `graphingest/resolution/README.md` | `34cb48010fe0755f2277329ae27a0219370b9b464d575d88ff485572095a716b` |
| `graphingest/resolution/contracts.go` | `376230d16a12345db2a62f93403077200178cd889e7f5262f2f6ee26eb53aa4e` |
| `graphingest/resolution/history/README.md` | `82c09af909c640ac54c40222ac0b2069610105996f1cf310493d24fae1933088` |
| `graphingest/resolution/history/scaling_benchmark_test.go` | `f542b5f9f66305f02932fb24659f72c9c7f6d60443a12e2700cab6132f3a7546` |
| `graphingest/resolution/scaling.md` | `4e0815b85334e666539aa361708635c1e621fc8f202ba10849713690de8b4741` |
| `graphingest/resolution/scaling_benchmark_test.go` | `8cc8ee39c6779d1e1c251e1f4e64b6a8b72791f421323dc3bfcf8b75744f21a9` |
| `recipe/graphexpand/README.md` | `d827f6377bde0b385e0859ee8cc44573d67a662f149eb4b3c66b4f8f80133435` |
| `recipe/graphsummary/README.md` | `e3cb33d76cc562829d25dc99cbb919d8ccf1376094636cb0dfa9844fdfa4f03a` |
| `recipe/graphsummary/citation_policy_test.go` | `f83d35bf0b198156b0e2545a9f71a037b1716cea3a1b86390e5085eef76f83f0` |
| `recipe/graphsummary/contracts.go` | `11a53e9cb4ca5935392534bb1629664205a8e2a15c6dda13965f40fe157a5bcf` |
| `recipe/graphsummary/scaling_benchmark_test.go` | `f402a617ed55699416c4db8fd83b09770b6d1b0cbeff5af206c7e7d2be1dcbd1` |
| `recipe/graphsummary/summary.go` | `d497f5eb7c0a8c0045e491c789b4a08f3953bc223c35873a923740834af13b9b` |
