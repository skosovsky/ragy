# Portable filter contract

A finalized schema admits only declared string, bool, int64 and finite float64
attributes. Whole nil/empty maps and omitted optional fields are valid; present
null, wrong-kind and nonfinite values are invalid. MatchCondition/MatchIR take
already admitted schema-normalized lookup values; they are not admission APIs for
arbitrary user maps.

For omitted fields, Eq/In/order are false and Neq is true. NOT/AND/OR compose
ordinary two-valued booleans; an empty read condition is true. Empty destructive
filters are rejected by storage adapters. These semantics are independent of SQL
NULL behavior. The portable conformance corpus checks all four kinds, omission and
nested logic, including adjacent exact integers above 2^53.

Public schema constructors return built-in typed fields. Generic scalar constraints
match those exact built-in kinds; named underlying-type constructors are not
supported. Ordinary Go aliases of built-in types retain their identical type.
No speculative named-type normalization or generic alias constructor is added.
NewBuilder requires a finalized schema; nil/zero/unfinalized builder states fail
ErrInvalidArgument. Final schema validation checks field names and kinds.
