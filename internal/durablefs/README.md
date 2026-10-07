# Local durable filesystem helpers

These private helpers implement narrow local Darwin/Linux filesystem mechanics.
The host supplies a trusted preprovisioned root with durable ancestors. Bounded
reads, file fsync and immediate-directory fsync support tested process restart and
process-crash recovery. They do not establish MkdirAll ancestor durability, hostile
filesystem isolation, network filesystem locking or hardware power-loss behavior.

Lock acquisition uses nonblocking flock and returns lifecycle conflict on
contention. Callers own and close the returned unduplicated handle. On Linux its
lock is attached to the open file description; duplicated descriptors would extend
that lifetime and are outside this helper's caller contract.

QueryPayloadError strips path and host-reader error text. Cancellation/deadline
and protocol sentinels remain identifiable; ordinary storage errors become
unavailable. Dense and tensor keep their identity, inventory, retirement and
publication checks visible in their own adapters. Shared mechanics are not a
replacement for those checks.
