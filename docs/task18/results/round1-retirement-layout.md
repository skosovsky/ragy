# Current-format retirement layout comparison

Same confirmed-cleaned old/tombstone-owner pair fixture in the current implementation. Before/after here means inventory compaction layout; it is separate from the original active-publication baseline. Maintenance is excluded from measured ordinary CAS. Preserved identities, exact reference fences and cleanup receipts remain in snapshot storage.

| Profile | Samples | Median ms/op [min–max] | Median B/op | Median allocs/op | Snapshot bytes |
|---|---:|---:|---:|---:|---:|
| CASBeforeCompaction/Artifacts32/History10 | 3 | 13.43 [12.81–15.86] | 1,622,923 | 2,172 | 119,747 |
| CASAfterCompaction/Artifacts32/History10 | 3 | 13.49 [12.93–14.12] | 919,068 | 5,398 | 43,137 |
| CASBeforeCompaction/Artifacts32/History100 | 3 | 37.78 [37.08–39.53] | 22,015,856 | 22,706 | 1,202,986 |
| CASAfterCompaction/Artifacts32/History100 | 3 | 22.21 [21.5–23.28] | 10,362,849 | 54,116 | 431,126 |
| CASBeforeCompaction/Artifacts32/History1000 | 3 | 510.4 [505.3–516.9] | 207,790,688 | 206,807 | 12,102,886 |
| CASAfterCompaction/Artifacts32/History1000 | 3 | 316.9 [316–319.2] | 118,497,416 | 535,401 | 4,320,926 |
