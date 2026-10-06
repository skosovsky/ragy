# Typed resolution history

Resolution results now include one `EntityDecision` per input entity and one
`RelationDecision` per input relation, in input order. Entity traces retain the
explicit policy decision, local mention ID, canonical ID (absent for ambiguity)
and original supports. Relation traces retain resolved endpoint IDs, state, the
host relation key (absent if endpoints are ambiguous) and original supports.
Grouping still keeps conflicting variants independently and selects no winner.

The optional JSON persistence profile captures `Record[TKind, TRel, TAttr]`:
host run/extraction configuration identity, optional predecessor snapshot ID,
complete typed input and resolver result, including variants and per-mention
decisions. Use the actual resolver output. This is a record of declared host policy
outcomes; capture does not rerun identity/equivalence policies or attest their domain
correctness. Policy and ontology identities identify the configuration; recomputation
must supply new identities when the host policy changes. All BYOT values in this
profile must support faithful, deterministic JSON serialization/deserialization.
Custom JSON methods must be pure, bounded, concurrency-safe and own decoded state.
Interfaces decode with `UseNumber`; persisted attribute schemas remain host-owned.

`Capture` validates trace cardinality/local mention/support associations, support
inventory and finite byte/fact/support limits. It authorizes every original locator
under the supplied binding before serialization; pinned publication membership and
freshness are checked separately from host retained-source admission. Results cannot
introduce a locator absent from the captured input. Serialized bytes are private;
each `Record()` or `Reference()` call returns independent state. Snapshot values
represent already-admitted owned data, like a returned retrieval result; fresh storage
reads always reauthorize against current host policy.

## Filesystem reference implementation

`NewFileStore` takes a host-owned root, byte/support limits and mandatory original
source admission. On supported local filesystems, `Append` writes and synchronizes a
temporary payload, publishes it with an atomic no-overwrite hard link and synchronizes
the directory. The host must durably prepare the root/ancestor directories and use a
filesystem supporting these primitives; remote/distributed durability is not claimed.
Only the temporary file created by that append is removed. Unknown records/staging
files are untouched. Identical concurrent appends are idempotent; mismatching existing
payloads fail. No latest pointer, deletion, scheduler or implicit retention policy is
introduced. The host controls retention of original source revisions and histories.

Record ID is the content digest. The storage filename also binds the complete support
inventory, preventing an allowed forged inventory from addressing an existing private
payload by ID alone. Reads authorize the complete locator list before payload I/O,
bound payload size, verify digest and exact decoded support inventory, and recheck
freshness after decoding. Revoked/deleted source evidence is denied unless the host
explicitly authorizes retained historical revisions under an appropriate binding.

After an append error/cancellation near the atomic link, the record may already exist.
Retain the original snapshot reference and inspect explicitly; do not silently rerun
resolution or create a replacement run. `Metadata.Parent` records the host-declared
predecessor; it does not prove predecessor availability or impose an ordered CAS log.
The host retains references in its durable source catalog and coordinates graph
materialization/publication using existing lifecycle contracts. A successful history
append is not evidence that graph publication succeeded.

Declared identities follow the resolver UTF-8 domain: run/extraction configuration,
input IDs/names/endpoints, result ontology/policy, group and per-mention decisions
are checked before authorization/serialization and during decoded inventory admission.
Required strings are nonempty; absent namespace/ambiguous fields remain explicitly
optional. `Metadata.Parent` remains an optional canonical SHA256 snapshot ID.
Malformed direct identities/configuration return `ragy.ErrInvalidArgument`, malformed
resolution decision/result identities return `ragy.ErrProtocol`, with an empty snapshot.
Raw persisted JSON must itself be valid UTF-8. BYOT kind/attribute codecs still must
serialize faithfully; core does not normalize or interpret host types. Existing valid
records retain their bytes/digests. Previously repaired malformed legacy identities
cannot be reconstructed from replacement characters: quarantine or migrate them under
host policy. Valid U+FFFD is preserved as a real identity character.

## Capacity and platform

The same `maxSupports` limit has several independent uses: each input entity/relation
list and each result entity/relation/unresolved list must fit it; each group variant
list must fit it; all input support occurrences together must fit it; and all result
variant/unresolved support occurrences together must fit it. These two occurrence
budgets are separate. Repeated locators consume capacity before inventory union.
Per-mention decision counts must exactly match the corresponding input counts.
This is not a set of independently configurable fact/variant quotas.

`maxBytes` bounds accepted JSON after Marshal has allocated the payload, not peak
heap, codec CPU or transient copies. Host must pre-bound BYOT values and supply
pure, bounded, faithful, concurrency-safe JSON methods that own decoded state.
Snapshot and Record code is portable; NewFileStore and filesystem operations are
available only on darwin/linux. MkdirAll does not prove ancestor durability; host
must provision the durable root. Unknown append requires inspection with the
original Reference. No automatic retry, record expiry or deletion is performed.

Stable-order inventory unions and subset validation can be quadratic in admitted
support counts. [Scaling measurements](../scaling.md) document reference workloads;
JSON cloning remains for ownership, without a performance claim.
