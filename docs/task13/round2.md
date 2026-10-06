# TASK-13 review round 2 — rejected

Candidate: `3303e82a79049d285ab796272635dda6bc9385d0f872595ecf2e11c38b0f8b5e`.

Completeness reviewer: 18/18 (100%), all prior gaps closed. Correctness reviewer: old defects closed, new P1 reproduced — candidate Search admission saw original TopK/FetchLimit instead of dispatched CandidateBudget. Candidate rejected despite completeness acceptance.

Repair: one owned candidate request constructor is used by both admission and dispatch; target admission clears dense vector as target dispatch does. Added a host budget-veto regression asserting no candidate/target payload call.

Both independent reviewers must accept the new identical candidate before commit.
