# Persistent dense profile

The dense adapter uses trusted local Darwin/Linux storage with retained catalogs
and exact publication/inventory admission. Published ready candidates must match
all pinned identity fields; retired, tombstoned and unpublished manifests are
excluded, and ambiguous live matches fail unavailable. Retired exclusion is an
explicit selector guard, not evidence of a previously confirmed native payload
bypass.

The host preprovisions the root and durable ancestors. The local profile verifies
process restart and process-crash recovery, not hardware power-loss durability.
File/root fsync does not establish MkdirAll ancestor durability. Root ownership is
a host boundary, not a hostile-path or symlink sandbox. Nonblocking local flock
returns contention without retry; close unduplicated handles and do not assume
network-filesystem or distributed locking.

The [durablefs contract](../../internal/durablefs/README.md) describes narrow shared
mechanics and redacted query failures. Dense/tensor catalog, cleanup and admission
checks remain type-specific and visible. No generic storage engine, clone/fence
consolidation or performance improvement is claimed by this change.
