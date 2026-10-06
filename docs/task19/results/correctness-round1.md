# TASK19 independent correctness round 1

Candidate `b9f26fd0398f4d5fa35ed91bd884cce69fb298ac7499a8b0ee54d3e3c9c20917`: **REJECT**. All 42 candidate file hashes matched at initial inspection. Runner tensor expected only tensor-candidate-maxsim, but frozen consumer emitted baseline and tensor profiles (108 rows). Actual report fails the required grid/strategy gate. A new candidate and final all-module run are required.

All graded Recall, MRR, nDCG, retrieved/delivered source-ID coverage and no-answer/failure denominators independently recomputed in Python matched all actual holdout reports. Freeze digests matched the published 536-file freeze. Text and graph mechanical gates passed. Explicit scope, forged citation, context overflow, candidate dispatch overflow and unknown accounting mutations were rejected. Preflight bugs in failed abstention, unsupported version, duplicate candidates and bounded dispatch accounting were repaired in source and exercised by focused race tests.

Focused same-candidate race: internal/task19 passed1.480s, recipe_comparison passed44.398s, final_contract passed2.764s. Runner five selftests passed, but lacked coverage for the actual tensor baseline grid. No implementation/qrel/contracts changes made by reviewer. Holdout first read only after candidate and authorization; no strategy tuning performed.
