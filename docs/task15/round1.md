# TASK-15 round 1 — rejected

Candidate `c0c4a623cb82a4a9d4788acedca2df1f25dc86774877f465d36316d645c18e27`.

Independent completeness: 11/13 = 84.62%. E04/E13 fail: recipe negative fixture omits mandatory Space, hiding the intended unsafe-reuse rejection behind invalid-argument; full core race fails. OTel Space comments misplaced (3 lint errors), resilience dimension literal (1 lint error). Other matrix rows confirmed.

Independent correctness: rejected, P2. Gemini multimodal propagates core Part.Validate error formatting arbitrary unknown Part.Kind verbatim. Unbounded unknown Kind can allocate a large error and disclose raw input despite MaxInputBytes. Reproducer uses PRIVATE_RAW_INPUT_MARKER; no IO occurs. Fix requires fixed sanitized local rejection before formatting plus adversarial regression.

Both reviewers verified all hashes before/after; no source modified during review. Remaining focused/provider/store races passed; no live-provider execution claimed.
