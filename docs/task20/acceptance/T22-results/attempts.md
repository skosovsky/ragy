# T22 failed setup attempts

- clean-consumer-cli-failed.log: initial invocation incorrectly used --source-sha; tracked helper requires positional reviewed_source. No consumer check ran. Correct exact-SHA invocation subsequently PASS.
- postgres-path-failed.log: explicit PATH omitted /usr/local/bin/docker, so isolation-label probes could not execute Docker. Container label was correct; fixed PATH.
- postgres-database-failed.log: default image initialized postgres DB, but established fixture bridge expects ragy DB. Created ragy and vector extension only in owned disposable container, then repeated all profiles.

No failed setup counted as PASS. Runtime implementation unchanged.
