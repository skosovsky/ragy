# Retained source access and indexing

Reference identifies one immutable namespace/source/revision/transformation/access/
artifact/representation. Locator adds byte or geometry selectors. Structural
validation never authenticates actual bytes. Host owns blob retention, domain
authorization, payload codecs and exact original representations.

Reader separates authoritative thin Catalog from admitted Loader. It captures
references, admits the complete batch before loading payload, clones owned payload
and rechecks identity/freshness. Denied or unavailable retained revisions fail
closed, without substituting latest. documents.Hydrator and layout.Resolver are
focused specializations. documents.RawStore is explicit unscoped administration,
with no scope/publication/retention guarantees.

A typical host flow is:

1. Catalog admits exact retained source references under the original Binding.
2. Loader obtains the authorized text/layout snapshot; clone ports deeply own BYOT.
3. layout.Project consumes an already authorized Document and checks binding
   freshness. It cannot prove that arbitrary supplied document belongs to that scope.
4. Splitter slices mapped original source ranges. Chunk metadata stays borrowed
   immutable; synthetic context is separate.
5. ProjectDocuments constructs the complete batch or no payload. Host indexes
   IndexText and retains original Document/SourceMapping, with explicit versioned IDs.
6. Retrieval exports original locators; Reader/Hydrator/Resolver reauthorizes those
   exact references and resolves retained originals. A mapping hash is not authority.

Runnable deterministic profiles (no paid model/network calls):

```sh
go test -race -count=1 ./documents -run TestChunkProjectionIndexAndRetainedResolution
go test -race -count=1 ./layout -run 'TestProjection|TestOCR|TestResolve'
```

[Retained chunk/index test](../documents/chunking_integration_test.go) uses actual
BM25, mapped splitting/projection and admitted Catalog/Loader hydration. A newer
revision never replaces old offsets; deletion/revocation prevents payload loading.
[Layout profiles](../layout/resolve_test.go) exercise scoped retained text/region/
cell/image resolution; [projection/OCR](../layout/project_test.go) preserves partial
coverage and explicit ImageText authority. The real PDF adapter profile is
[separate](../adapters/pdf/pipeline_test.go) and requires its supported runtime.

## Mapping wire, exact coordinates and ancestry

MappedText owns rendered text and fragment/support slices. OriginalText validates
exact UTF-8 source spans. DerivedText/SupportedOriginalText retain support-only
precision. Slice retains the original broad support ancestry plus narrowed exact
Location: Supports is not a list of only precise displayed quotations. Joining
originals retains their independent source coordinates.

The mapping JSON profile uses ordinary encoding/json with unknown-field/trailing
value rejection and structural Validate. Duplicate members use Go decoder's later
value semantics, case-insensitive field aliases are accepted, missing zero fields
may be valid, and malformed Unicode escapes are repaired before validation. This
differs from strict locator/evidence envelopes; it is not an authentication or
canonical untrusted-wire boundary. Store only MarshalJSON output through trusted
authenticated storage when requiring canonical mapping bytes. Decoding preserves
the previous snapshot on failure and owns support slices on success.

An attacker can declare a structurally exact mapping containing false quote text.
External citation trust requires authorized retained Resolve; declared precision,
serialization and digests cannot certify source authenticity. DecodeLocator's
strict shape likewise validates geometry rather than permission or actual content.

Repeated standalone OriginalText calls still scan UTF-8 source bytes for every
call. OriginalTexts checks one retained snapshot once, then validates each locator
and byte boundary, returning an ordered owned batch or no payload on error.
layout region resolution uses this batch. MappedText.Validate checks rendered UTF-8
once before its fragment boundaries. [Before/after measurements](scaling.md) show
the measured long-page workload and remaining costs. Host must bound source bytes, fragment/
support populations and callback work; finite count/wire caps do not bound CPU/RSS.
