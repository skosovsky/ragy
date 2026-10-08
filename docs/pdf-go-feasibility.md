# Pure-Go PDF replacement: blocked

Evaluated 2026-10-08 against the existing `adapters/pdf/testdata/manual.pdf`.
The Python adapter and its public contract remain unchanged. This is the explicit
PDF exception to the tooling migration, not a successful replacement.

| Candidate | Exact source | Observed mismatch |
|---|---|---|
| github.com/razvandimescu/gopdf | 2660192c5d162855418503ad80ef32e2e8f14d77 | Page 0 `Tables()` returns an empty slice for the existing bordered/merged-cell table. TextSpans are line-sized rather than the existing normalized words. |
| github.com/ledongthuc/pdf | 6c8c28e0e8a07a3452f9d8cf3090ed6e856d8f24 | Page 1 `Content()` returns every Helvetica glyph at the same run X coordinate with W=0; no table cell grid is returned. |

Neither candidate exposes context-aware parsing. A context check around a
synchronous parse does not preserve the existing deadline guarantee; returning
from a goroutine on timeout would leave parser work running. Both fail the gate
before full labels, image geometry and adversarial limit coverage can be accepted.
No claim is made that all other functionality was tested or is impossible.

Reproduction: checkout the exact candidate revisions, open the fixture through
`pdf.OpenFile` and call `Page(0).TextSpans()` / `Page(0).Tables()` for gopdf;
for ledongthuc open with `pdf.Open` and call `Page(1).Content()`. The retained
fixture assertions in the adapter define expected cell geometry and word spans.

A replacement now requires an owner decision: broaden backend constraints,
explicitly change the PDF contract, or fund parser-level implementation beyond
this migration. Tooling can be used without Python; actual PDF integration still
requires the existing engine until that decision. Full Python removal is pending.

The exact probes are retained under `pdf-feasibility/*.go.txt`; copy each into an
isolated module with the corresponding pinned dependency and pass the fixture path
as argv[1]. `pdf-feasibility/results.txt` records the observed output summary.
