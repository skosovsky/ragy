# Whole-process resource costs

Recorded by Darwin `/usr/bin/time -l`. CPU seconds and maximum RSS include fixture creation/publication, filesystem synchronization, cleanup and any compiler work. They are process costs, not per-query budgets; raw platform RSS values are retained without cross-platform normalization. Repeated N10k dense staging dominates elapsed process duration but is outside query timers.

| Profile | Wall seconds | User CPU seconds | System CPU seconds | Maximum RSS (raw platform units) |
|---|---:|---:|---:|---:|
| serial-after-core | 12.00 | 13.61 | 2.26 | 161890304 |
| serial-after-history | 6.99 | 3.48 | 1.07 | 148668416 |
| serial-after-managed | 28.45 | 30.73 | 3.03 | 158580736 |
| serial-after-persistent | 299.93 | 13.93 | 27.87 | 156549120 |
| serial-before-core | 29.29 | 28.38 | 1.77 | 161579008 |
| serial-before-history | 5.40 | 2.94 | 1.01 | 148094976 |
| serial-before-managed | 113.12 | 73.70 | 3.34 | 154976256 |
| serial-before-persistent | 356.05 | 13.88 | 27.22 | 161267712 |
| serial-current-retirement | 12.29 | 10.21 | 1.56 | 148144128 |
