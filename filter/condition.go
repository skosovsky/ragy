package filter

// Condition is a validated filter tree produced only by Builder.Build().
// A zero-value Condition means "no filter" in option structs such as RetrieveOptions.
// Adapters read the tree through IR(); callers must not assemble IR manually.
type Condition struct {
	ir IR
}

// IR exposes the validated filter tree for adapter translation.
func (c Condition) IR() IR {
	if c.ir == nil {
		return emptyExpr{}
	}
	return c.ir
}

// ValidateCondition checks that a condition is valid against schema rules when applicable.
func ValidateCondition(c Condition) error {
	return ValidateIR(c.IR())
}

func emptyBuiltCondition() Condition {
	return Condition{ir: emptyExpr{}}
}

func conditionFromIR(ir IR) (Condition, error) {
	if err := ValidateIR(ir); err != nil {
		return Condition{}, err
	}
	return Condition{ir: ir}, nil
}
