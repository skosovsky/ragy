# TASK-13 review round 1 — rejected

Independent reviewers: `/root/task13_completeness` and `/root/task13_correctness` (no implementation participation).

Completeness: 15/18 = 83.33%. Open: C04 cross-type wrapper-dependent admission, C09 actual missing retained revision through decorated reads including a prior read, C18 failed lint.

Correctness: P1 tensor Search did not invoke child RequestReadAdmission; P2 cross-type projection depended on presence of a forwarding wrapper interface. Both independently reproduced in `/tmp/ragy-review13/repro_test.go`; root source unchanged during review.

Repairs: uniform explicit AdmissionProject contract for cross-type scoped/pinned requests and all consumers; recursive Search admission and coverage merging; pins/partial reads bypass cache storage to preserve physical-retention checks at leaf; real joint persistent cleanup regression; lint fixes. Repeat both independent reviews on the repaired candidate.
