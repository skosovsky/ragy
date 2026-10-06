# Whole-process resource costs

Measurement source lineage: final CAS/history rows were rerun on round2 `70b27da85229b46fde37546f2c06a4f46256d2dbb4ffb90181608d4b7e322276` (candidate `af0961f3fbea174049fd61b504d85571068921652880439fb04b43ba9ba1e99f`). After core/managed/persistent query rows remain measured on round1 `4ecff6fadf7faaa5e59baf3619e9bfc67b1e8227ce19c9b6f70f07a00c83f431`; their timed production paths are unchanged by the final CAS batch guard. Baseline is `2f76506e128d6831d2406dd25d96ef2d73ee570b85b612f0ac26ac65c6f69997`. Their retained whole-process CPU/RSS includes round1 untimed CAS setup and is attributed to round1, not to the final whole process. See measurement-validation.txt and source columns in the CSVs.

Recorded by Darwin `/usr/bin/time -l`. CPU seconds and maximum RSS include fixture creation/publication, filesystem synchronization, cleanup and any compiler work. They are process costs, not per-query budgets; raw platform RSS values are retained without cross-platform normalization. Repeated N10k dense staging dominates elapsed process duration but is outside query timers.

| Profile | Wall seconds | User CPU seconds | System CPU seconds | Maximum RSS (raw platform units) |
|---|---:|---:|---:|---:|
| serial-after-core | 12.00 | 13.61 | 2.26 | 161890304 |
| serial-after-history | 6.42 | 4.08 | 1.27 | 152387584 |
| serial-after-managed | 28.45 | 30.73 | 3.03 | 158580736 |
| serial-after-persistent | 299.93 | 13.93 | 27.87 | 156549120 |
| serial-before-core | 29.29 | 28.38 | 1.77 | 161579008 |
| serial-before-history | 5.40 | 2.94 | 1.01 | 148094976 |
| serial-before-managed | 113.12 | 73.70 | 3.34 | 154976256 |
| serial-before-persistent | 356.05 | 13.88 | 27.22 | 161267712 |
| serial-current-retirement | 12.03 | 9.30 | 1.23 | 149307392 |
