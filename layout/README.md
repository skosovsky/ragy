# Layout source addressing

Each `Document.Pages[i].Reference` uniquely addresses the retained normalized text
of one physical page. Reusing an exact reference is `ragy.ErrInvalidArgument`, even
for equal/empty text, different byte lengths, different geometry or partial coverage.
`Document.Validate` enforces this; `Project` validates the complete document before
any `ImageText` callback and returns no projection on failure. Text citations use
exact reference plus byte span. Adding physical page to an ID hash would leave an
ambiguous loader address; location hashing is unchanged.

Distinct page references let the admitted Catalog → Loader path resolve original
text independently, without latest-version fallback. Page.Reference and geometry
alone do not attest that arbitrary supplied text belongs to a read scope. Project
requires an already authorized document; Binding checks freshness.

Cells/images use complete Locators with physical geometry and cell/region selectors.
Distinct selectors may share the same retained whole artifact if the host's loader
independently resolves every selector to that artifact's immutable original content.
Duplicate logical cells or complete image locators within a page remain invalid.
Cross-page selectors remain distinct; sharing never relaxes unique normalized page
text references. OCR/image descriptions remain derived from their original locators.

The built-in `layout.Resolver` consumes one `Retained.Original` record per reference;
its cell/image profile expects individually retained originals. Validation/projection
of whole-artifact sharing does not certify that profile as a whole-artifact selector
loader. A host using such sharing must supply a loader with the required selectors.
Parsers, retained storage, authorization and media rendering remain host-owned.
