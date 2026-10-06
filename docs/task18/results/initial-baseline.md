# Initial overlapping baseline (not final comparison)

These raw captures overlapped other benchmark processes. Observed latency ranges are retained for evidence and are not the final before/after timing comparison. N=100/1000/10000 and retained history10/100/1000; see workload.md. ns/op measures wall time.

| Profile | Samples | Median ms/op [min–max] | Median B/op | Median allocs/op | Snapshot bytes |
|---|---:|---:|---:|---:|---:|
| Lexical/Bulk/N100 | 3 | 0.6203 [0.6076–0.6226] | 164,920 | 932 |  |
| Lexical/Read/unique0/N100 | 3 | 0.06283 [0.06008–0.1002] | 86,504 | 536 |  |
| Lexical/Read/common/N100 | 3 | 0.1388 [0.12–0.1839] | 105,003 | 575 |  |
| Lexical/Update/N100 | 3 | 0.01292 [0.01205–0.01331] | 2,056 | 13 |  |
| Lexical/ConcurrentReadUpdate/N100 | 3 | 0.05762 [0.05266–0.06116] | 77,710 | 484 |  |
| Lexical/Bulk/N1000 | 3 | 11.1 [10.72–11.21] | 1,517,788 | 7,559 |  |
| Lexical/Read/unique0/N1000 | 3 | 0.634 [0.6053–0.6914] | 697,398 | 3,447 |  |
| Lexical/Read/common/N1000 | 3 | 1.56 [1.016–2.293] | 855,860 | 3,498 |  |
| Lexical/Update/N1000 | 3 | 0.02767 [0.02716–0.03354] | 1,808 | 11 |  |
| Lexical/ConcurrentReadUpdate/N1000 | 3 | 0.3807 [0.3394–0.4854] | 626,664 | 3,106 |  |
| Lexical/Bulk/N10000 | 3 | 557.2 [547.4–639.7] | 14,431,784 | 71,346 |  |
| Lexical/Read/unique0/N10000 | 3 | 6.192 [5.358–6.397] | 6,275,641 | 30,560 |  |
| Lexical/Read/common/N10000 | 3 | 7.92 [7.798–7.981] | 7,571,133 | 30,671 |  |
| Lexical/Update/N10000 | 3 | 0.1532 [0.1527–0.1706] | 1,792 | 11 |  |
| Lexical/ConcurrentReadUpdate/N10000 | 3 | 2.692 [2.578–3.042] | 5,706,130 | 27,782 |  |
| Filestore/CAS/History10 | 3 | 22.77 [21.19–26.54] | 97,535 | 278 | 8434.0 |
| Filestore/CAS/History100 | 3 | 26.39 [25.16–32.53] | 955,514 | 2,055 | 83854.0 |
| Filestore/CAS/History1000 | 3 | 48.16 [43.07–56.33] | 12,422,474 | 18,867 | 844354.0 |
| ManagedLexical/N100/Scope100 | 3 | 2.065 [2.003–2.127] | 1,706,168 | 11,880 |  |
| ManagedLexical/N100/Scope10 | 3 | 1.287 [1.157–3.46] | 746,610 | 4,435 |  |
| ManagedLexical/N1000/Scope100 | 3 | 28.92 [27.22–42.22] | 16,760,762 | 115,782 |  |
| ManagedLexical/N1000/Scope10 | 3 | 12.09 [9.073–17.82] | 6,762,771 | 41,669 |  |
| ManagedLexical/N10000/Scope100 | 3 | 669.4 [663–674.5] | 162,403,488 | 1,152,211 |  |
| ManagedLexical/N10000/Scope10 | 3 | 77.43 [75.88–78.61] | 64,192,392 | 413,489 |  |
| ManagedGraph/N100/Scope100/Admission | 3 | 0.0006739 [0.0006658–0.000688] | 80 | 3 |  |
| ManagedGraph/N100/Scope100/FindByIDsFullView | 3 | 1.284 [1.274–1.331] | 768,381 | 1,917 |  |
| ManagedGraph/N100/Scope100/TraverseDepth4 | 3 | 1.285 [1.28–1.315] | 774,515 | 1,954 |  |
| ManagedGraph/N100/Scope10/Admission | 3 | 0.0006737 [0.0006696–0.0006753] | 80 | 3 |  |
| ManagedGraph/N100/Scope10/FindByIDsFullView | 3 | 0.8899 [0.8746–0.8937] | 669,916 | 1,531 |  |
| ManagedGraph/N100/Scope10/TraverseDepth4 | 3 | 0.9116 [0.8942–0.9444] | 676,331 | 1,568 |  |
| ManagedGraph/N1000/Scope100/Admission | 3 | 0.0006898 [0.0006863–0.0006937] | 80 | 3 |  |
| ManagedGraph/N1000/Scope100/FindByIDsFullView | 3 | 42.43 [42.08–42.84] | 7,933,482 | 18,256 |  |
| ManagedGraph/N1000/Scope100/TraverseDepth4 | 3 | 42.68 [42.23–42.87] | 7,939,813 | 18,293 |  |
| ManagedGraph/N1000/Scope10/Admission | 3 | 0.0006795 [0.0006723–0.0006913] | 80 | 3 |  |
| ManagedGraph/N1000/Scope10/FindByIDsFullView | 3 | 10.38 [10.19–10.5] | 6,706,403 | 14,612 |  |
| ManagedGraph/N1000/Scope10/TraverseDepth4 | 3 | 10.34 [10.02–10.35] | 6,712,351 | 14,647 |  |
| ManagedGraph/N10000/Scope100/Admission | 3 | 0.0007712 [0.0006782–0.001119] | 80 | 3 |  |
| ManagedGraph/N10000/Scope100/FindByIDsFullView | 3 | 3551 [3486–4117] | 76,348,304 | 181,208 |  |
| ManagedGraph/N10000/Scope100/TraverseDepth4 | 3 | 3479 [3469–3584] | 76,354,640 | 181,245 |  |
| ManagedGraph/N10000/Scope10/Admission | 3 | 0.0007162 [0.0007026–0.000819] | 80 | 3 |  |
| ManagedGraph/N10000/Scope10/FindByIDsFullView | 3 | 299.8 [270.3–528] | 65,441,080 | 144,974 |  |
| ManagedGraph/N10000/Scope10/TraverseDepth4 | 3 | 274.7 [263.5–280.7] | 65,447,400 | 145,011 |  |
| DenseExactScan/N100 | 3 | 6.126 [5.809–9.654] | 1,839,720 | 9,349 |  |
| DenseExactScan/N1000 | 3 | 82.59 [69.42–88.65] | 19,744,440 | 92,541 |  |
| DenseExactScan/N10000 | 3 | 772 [741.5–920] | 198,080,264 | 922,485 |  |
| TensorCandidateMaxSim/N100/Candidates10Pct | 3 | 2.098 [1.926–2.162] | 1,063,222 | 3,884 |  |
| TensorCandidateMaxSim/N100/Candidates100Pct | 3 | 6.61 [6.432–7.276] | 2,070,427 | 11,228 |  |
| TensorCandidateMaxSim/N1000/Candidates10Pct | 3 | 27.29 [26.95–34.07] | 11,628,260 | 36,623 |  |
| TensorCandidateMaxSim/N1000/Candidates100Pct | 3 | 118 [80.61–145.1] | 21,805,160 | 109,742 |  |
| TensorCandidateMaxSim/N10000/Candidates10Pct | 3 | 540.8 [326.4–699.3] | 110,193,792 | 363,138 |  |
| TensorCandidateMaxSim/N10000/Candidates100Pct | 3 | 2725 [2613–3640] | 213,982,504 | 1,092,783 |  |
| Filestore/CAS/Artifacts1/History10 | 3 | 22.8 [22.45–23.99] | 97,531 | 278 | 8434.0 |
| Filestore/CAS/Artifacts1/History100 | 3 | 24.32 [24.29–28.28] | 934,873 | 2,048 | 83854.0 |
| Filestore/CAS/Artifacts1/History1000 | 3 | 38.5 [37.67–44.81] | 12,422,168 | 18,866 | 844354.0 |
| Filestore/CAS/Artifacts32/History10 | 3 | 31.14 [25.53–32.42] | 1,257,220 | 1,703 | 112724.0 |
| Filestore/CAS/Artifacts32/History100 | 3 | 119 [51.53–127.7] | 18,906,632 | 18,626 | 1132334.0 |
| Filestore/CAS/Artifacts32/History1000 | 3 | 228.3 [218.9–233.1] | 180,977,760 | 169,949 | 11390534.0 |
