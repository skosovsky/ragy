# TASK18 measured scaling

Measurement source lineage: final CAS/history rows were rerun on round2 `70b27da85229b46fde37546f2c06a4f46256d2dbb4ffb90181608d4b7e322276` (candidate `af0961f3fbea174049fd61b504d85571068921652880439fb04b43ba9ba1e99f`). After core/managed/persistent query rows remain measured on round1 `4ecff6fadf7faaa5e59baf3619e9bfc67b1e8227ce19c9b6f70f07a00c83f431`; their timed production paths are unchanged by the final CAS batch guard. Baseline is `2f76506e128d6831d2406dd25d96ef2d73ee570b85b612f0ac26ac65c6f69997`. Their retained whole-process CPU/RSS includes round1 untimed CAS setup and is attributed to round1, not to the final whole process. See measurement-validation.txt and source columns in the CSVs.

Medians of three samples; bracketed values are observed min–max. Wall-time measurements on local APFS, GOMAXPROCS4. See workload.md for corpus, admission and storage limits. Setup is excluded. These values do not establish a latency SLO or tail percentile.

| Profile | Before ms/op [min–max] | After ms/op [min–max] | Before → after B/op | Before → after allocs/op |
|---|---:|---:|---:|---:|
| Lexical/Bulk/N100 | 0.5825 [0.5821–0.5841] | 0.5393 [0.5387–0.5417] | 164,498 → 164,760 | 932 → 932 |
| Lexical/Read/unique0/N100 | 0.04643 [0.04634–0.04757] | 0.00183 [0.001827–0.001878] | 86,589 → 2,400 | 536 → 29 |
| Lexical/Read/common/N100 | 0.06521 [0.06518–0.06552] | 0.03919 [0.03831–0.03933] | 104,833 → 53,638 | 575 → 187 |
| Lexical/Update/N100 | 0.01146 [0.01133–0.01167] | 0.01025 [0.01023–0.01029] | 2,067 → 2,055 | 13 → 13 |
| Lexical/ConcurrentReadUpdate/N100 | 0.03087 [0.03084–0.03189] | 0.003172 [0.00317–0.003172] | 77,376 → 2,350 | 483 → 27 |
| Lexical/Bulk/N1000 | 10.43 [10.39–10.46] | 5.599 [5.593–5.609] | 1,528,980 → 1,520,626 | 7,561 → 7,560 |
| Lexical/Read/unique0/N1000 | 0.4774 [0.4539–0.4826] | 0.002064 [0.002043–0.002083] | 698,587 → 2,415 | 3,448 → 29 |
| Lexical/Read/common/N1000 | 0.7195 [0.6818–0.724] | 0.5244 [0.5113–0.526] | 858,570 → 601,360 | 3,499 → 1,125 |
| Lexical/Update/N1000 | 0.02665 [0.02647–0.02669] | 0.01077 [0.01057–0.01094] | 1,824 → 1,810 | 11 → 11 |
| Lexical/ConcurrentReadUpdate/N1000 | 0.2545 [0.2479–0.2682] | 0.003456 [0.003342–0.003485] | 625,207 → 2,329 | 3,108 → 27 |
| Lexical/Bulk/N10000 | 541.7 [541.3–542.7] | 59.11 [58.82–59.38] | 14,506,448 → 14,394,192 | 71,358 → 71,339 |
| Lexical/Read/unique0/N10000 | 4.843 [4.733–18.44] | 0.00189 [0.001869–0.001921] | 6,279,036 → 2,385 | 30,561 → 29 |
| Lexical/Read/common/N10000 | 15.23 [8.335–16.4] | 5.898 [5.738–5.959] | 7,568,046 → 5,083,584 | 30,670 → 10,330 |
| Lexical/Update/N10000 | 0.1533 [0.1496–1.267] | 0.01147 [0.01138–0.01157] | 1,792 → 1,795 | 11 → 11 |
| Lexical/ConcurrentReadUpdate/N10000 | 6.167 [2.31–6.424] | 0.003468 [0.003375–0.003573] | 5,727,918 → 2,322 | 27,903 → 27 |
| Filestore/CAS/Artifacts1/History10 | 11.31 [10.96–11.48] | 11.63 [11.59–12.04] | 97,533 → 116,814 | 278 → 348 |
| Filestore/CAS/Artifacts1/History100 | 12.47 [12.25–12.81] | 12.7 [12.11–14.49] | 986,846 → 1,172,224 | 2,055 → 2,487 |
| Filestore/CAS/Artifacts1/History1000 | 27.25 [27.1–33.5] | 29.21 [29.01–30.01] | 12,422,482 → 14,605,344 | 18,867 → 22,918 |
| Filestore/CAS/Artifacts32/History10 | 13.48 [12.71–13.61] | 12.77 [12.75–13.12] | 1,386,214 → 1,542,076 | 1,712 → 1,844 |
| Filestore/CAS/Artifacts32/History100 | 33.66 [33.14–38.37] | 34.71 [33.96–35.56] | 18,906,124 → 20,736,680 | 18,624 → 19,655 |
| Filestore/CAS/Artifacts32/History1000 | 189.1 [187.7–191] | 225.8 [212.8–232.5] | 180,977,240 → 199,640,256 | 169,947 → 179,997 |
| ManagedLexical/N100/Scope100 | 1.9 [1.9–1.916] | 1.466 [1.455–1.507] | 1,705,473 → 1,379,974 | 11,880 → 4,366 |
| ManagedLexical/N100/Scope10 | 0.9814 [0.8435–1.02] | 1.175 [1.157–4.976] | 748,041 → 1,140,964 | 4,434 → 1,286 |
| ManagedLexical/N1000/Scope100 | 28.73 [25.09–28.92] | 14.29 [14.24–14.49] | 16,741,660 → 14,318,298 | 115,804 → 40,571 |
| ManagedLexical/N1000/Scope10 | 8.476 [7.973–8.488] | 11.65 [11.58–11.65] | 6,761,738 → 11,938,503 | 41,678 → 9,931 |
| ManagedLexical/N10000/Scope100 | 650.7 [640.2–668.9] | 131.7 [130.4–133] | 162,427,400 → 122,666,696 | 1,152,244 → 402,283 |
| ManagedLexical/N10000/Scope10 | 78.91 [73.7–103.7] | 96.99 [96.42–97.14] | 64,191,464 → 114,486,168 | 413,481 → 95,981 |
| ManagedGraph/N100/Scope100/Admission | 0.0006844 [0.0006801–0.0008229] | 0.0006491 [0.0006476–0.0006571] | 80 → 80 | 3 → 3 |
| ManagedGraph/N100/Scope100/FindByIDsFullView | 1.355 [1.328–1.397] | 0.9449 [0.9378–0.9479] | 767,986 → 870,189 | 1,916 → 2,150 |
| ManagedGraph/N100/Scope100/TraverseDepth4 | 1.506 [1.341–1.545] | 0.9565 [0.9539–0.96] | 774,441 → 876,959 | 1,954 → 2,188 |
| ManagedGraph/N100/Scope10/Admission | 0.0008235 [0.0007236–0.0008632] | 0.0006538 [0.0006535–0.0006555] | 80 → 80 | 3 → 3 |
| ManagedGraph/N100/Scope10/FindByIDsFullView | 2.479 [1.493–2.728] | 0.8558 [0.8462–0.8563] | 670,079 → 750,540 | 1,532 → 1,575 |
| ManagedGraph/N100/Scope10/TraverseDepth4 | 1.623 [1.227–1.81] | 0.8719 [0.869–0.8773] | 676,392 → 756,685 | 1,569 → 1,611 |
| ManagedGraph/N1000/Scope100/Admission | 0.0007592 [0.0007158–0.001283] | 0.000662 [0.000659–0.0006658] | 80 → 80 | 3 → 3 |
| ManagedGraph/N1000/Scope100/FindByIDsFullView | 59.83 [52.12–64.83] | 9.806 [9.782–9.855] | 7,933,344 → 9,531,211 | 18,256 → 20,329 |
| ManagedGraph/N1000/Scope100/TraverseDepth4 | 58.56 [42.63–71.96] | 9.888 [9.791–9.959] | 7,939,858 → 9,537,606 | 18,293 → 20,366 |
| ManagedGraph/N1000/Scope10/Admission | 0.0007669 [0.0007211–0.0007923] | 0.0006626 [0.0006537–0.0006679] | 80 → 80 | 3 → 3 |
| ManagedGraph/N1000/Scope10/FindByIDsFullView | 11.36 [11.14–11.6] | 8.651 [8.584–8.671] | 6,705,852 → 7,914,647 | 14,610 → 14,863 |
| ManagedGraph/N1000/Scope10/TraverseDepth4 | 13.71 [13.33–15.01] | 8.642 [8.568–8.727] | 6,712,665 → 7,920,982 | 14,649 → 14,900 |
| ManagedGraph/N10000/Scope100/Admission | 0.0006855 [0.0006852–0.000784] | 0.0006732 [0.0006642–0.0006743] | 80 → 80 | 3 → 3 |
| ManagedGraph/N10000/Scope100/FindByIDsFullView | 3698 [3597–3777] | 97.2 [96.55–97.24] | 76,348,192 → 89,248,692 | 181,208 → 201,515 |
| ManagedGraph/N10000/Scope100/TraverseDepth4 | 7305 [6840–9936] | 97.4 [95.72–98.42] | 76,354,656 → 89,255,036 | 181,246 → 201,552 |
| ManagedGraph/N10000/Scope10/Admission | 0.0009009 [0.0008521–0.0009979] | 0.0006655 [0.0006631–0.0006747] | 80 → 80 | 3 → 3 |
| ManagedGraph/N10000/Scope10/FindByIDsFullView | 370.2 [340.3–901] | 81.92 [81.91–82.08] | 65,440,952 → 75,298,400 | 144,974 → 147,161 |
| ManagedGraph/N10000/Scope10/TraverseDepth4 | 301.8 [287.6–759] | 84.84 [83.84–85.66] | 65,447,272 → 75,301,988 | 145,011 → 147,191 |
| DenseExactScan/N100 | 9.663 [5.78–16.09] | 4.796 [4.662–4.98] | 1,821,412 → 1,822,718 | 9,347 → 9,354 |
| DenseExactScan/N1000 | 178 [82.6–252.6] | 62.53 [59.52–87.57] | 19,573,312 → 19,570,960 | 92,582 → 92,555 |
| DenseExactScan/N10000 | 671.1 [605.1–1089] | 550.1 [496.8–1698] | 196,319,504 → 196,334,808 | 922,499 → 922,516 |
| TensorCandidateMaxSim/N100/Candidates10Pct | 1.69 [1.642–1.703] | 5.23 [2.015–5.436] | 1,061,338 → 1,061,735 | 3,882 → 3,883 |
| TensorCandidateMaxSim/N100/Candidates100Pct | 6.252 [6.183–6.267] | 5.988 [5.95–6.28] | 2,059,122 → 2,059,436 | 11,228 → 11,225 |
| TensorCandidateMaxSim/N1000/Candidates10Pct | 16.86 [16.74–17.09] | 16.45 [16.39–16.68] | 11,616,661 → 11,616,738 | 36,601 → 36,605 |
| TensorCandidateMaxSim/N1000/Candidates100Pct | 69.21 [68.99–70.63] | 59.8 [59.33–64.73] | 21,692,916 → 21,692,160 | 109,739 → 109,740 |
| TensorCandidateMaxSim/N10000/Candidates10Pct | 155.9 [154.4–159.7] | 254.4 [189.6–263.4] | 110,085,392 → 110,073,376 | 363,158 → 363,133 |
| TensorCandidateMaxSim/N10000/Candidates100Pct | 670 [661.6–681.5] | 659.3 [655.7–683.3] | 212,859,072 → 212,861,208 | 1,092,760 → 1,092,770 |
