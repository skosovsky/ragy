# Neo4j Runner bridge

The host supplies a typed Runner; this adapter does not construct Cypher or certify a remote driver. Retrieve supports unrestricted current reads only. Scoped reads and pinned publication reads return a protected ErrUnsupported before traversal. The host owns traversal execution and authorized input.

Every Retrieve return passes a final read delivery gate. Cancellation suppresses success, empty results and ordinary projection prefixes. When context cancellation coincides with an ordinary Runner/projection error, that already observed cause remains discoverable through errors.Is; the cancellation error text and errors.As do not expose the private callback payload. Independently supplied ProtectionError values retain the existing access.Protect cause contract. With a valid read, projection failures retain the validated node prefix and ErrProtocol; Runner failures return no documents. There is one traversal and no hidden retry. Nodes retain traversal rank and ScoreAbsent; BackendFetchLimit truncates traversal order.

Traverse/Upsert are explicit graph.Store administrative operations. This delivery change does not add scoped remote enforcement or alter their contract.
