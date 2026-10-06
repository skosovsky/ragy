# Current-format retirement layout comparison

Measurement source lineage: final CAS/history rows were rerun on round2 `70b27da85229b46fde37546f2c06a4f46256d2dbb4ffb90181608d4b7e322276` (candidate `af0961f3fbea174049fd61b504d85571068921652880439fb04b43ba9ba1e99f`). After core/managed/persistent query rows remain measured on round1 `4ecff6fadf7faaa5e59baf3619e9bfc67b1e8227ce19c9b6f70f07a00c83f431`; their timed production paths are unchanged by the final CAS batch guard. Baseline is `2f76506e128d6831d2406dd25d96ef2d73ee570b85b612f0ac26ac65c6f69997`. Their retained whole-process CPU/RSS includes round1 untimed CAS setup and is attributed to round1, not to the final whole process. See measurement-validation.txt and source columns in the CSVs.

Same confirmed-cleaned old/tombstone-owner pair fixture in the current implementation. Before/after here means inventory compaction layout; it is separate from the original active-publication baseline. Maintenance is excluded from measured ordinary CAS. Preserved identities, exact reference fences and cleanup receipts remain in snapshot storage.

| Profile | Samples | Median ms/op [min–max] | Median B/op | Median allocs/op | Snapshot bytes |
|---|---:|---:|---:|---:|---:|
| CASBeforeCompaction/Artifacts32/History10 | 3 | 14.8 [14.08–15.4] | 1,586,548 | 2,168 | 119,746 |
| CASAfterCompaction/Artifacts32/History10 | 3 | 11.89 [11.89–12.25] | 943,819 | 5,410 | 43,137 |
| CASBeforeCompaction/Artifacts32/History100 | 3 | 41.5 [40.39–52.49] | 22,015,856 | 22,706 | 1,202,986 |
| CASAfterCompaction/Artifacts32/History100 | 3 | 25.29 [24.11–51.22] | 10,362,354 | 54,115 | 431,126 |
| CASBeforeCompaction/Artifacts32/History1000 | 3 | 560.4 [550.2–657.2] | 207,790,688 | 206,807 | 12,102,886 |
| CASAfterCompaction/Artifacts32/History1000 | 3 | 324.1 [313.5–327.8] | 118,497,272 | 535,400 | 4,320,926 |
