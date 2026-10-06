# TASK-16 round 1 — rejected

Candidate `084fc66c8ba61688c5c52c5833cb6b67bba8db3f4dd2fe507c8d56e0eb34fc4d`.

Independent completeness: 11/13 = 84.62%. B05/B06 fail; other requirements confirmed including complete successful checks.

Independent correctness: rejected, P1. Text delivered coverage correlates selected documents and artifact snippets only by Document.ID. Valid BYOT resolver can produce different MergeKeys for documents sharing a storage ID. Two subquestions, same-storage-key documents with fact-one/fact-two content, assessor selects both; artifact default dedup retains one snippet, but ID-only correlation falsely marks both delivered and claims Complete.

Independent overlay reproducer held outside source; frozen hashes unchanged before/after both reviews. Core and six affected nested race suites exited 0; core lint 0 issues. Tests passing did not override this confirmed defect. Repairs must correlate actual retained input contributors and preserve snapshot/recording ownership.
