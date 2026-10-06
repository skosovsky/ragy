package filter

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// ValidateScopeProfile checks the reference scalar Eq/In/And mandatory profile.
// It is deliberately distinct from broader optional query-filter capabilities.
func ValidateScopeProfile(condition Condition) error {
	if err := ValidateCondition(condition); err != nil {
		return err
	}
	return validateScopeExpression(condition.IR())
}
func validateScopeExpression(expression IR) error {
	switch node := expression.(type) {
	case eqExpr, inExpr:
		return nil
	case andExpr:
		for _, child := range node.exprs {
			if err := validateScopeExpression(child); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("%w: mandatory predicate profile", ragy.ErrUnsupported)
	}
}
