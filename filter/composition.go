package filter

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// Intersect combines built conditions using AND after validating each against schema.
// Empty conditions are identities. Contradictory predicates remain an intersection
// matching no records; they are never replaced with an unrestricted condition.
// Conditions and their scalar values are immutable, so the result can be shared
// between concurrent requests without exposing mutable predicate storage.
func Intersect(schema Schema, conditions ...Condition) (Condition, error) {
	if err := schema.validateFinalized(); err != nil {
		return Condition{}, err
	}
	children := make([]IR, 0, len(conditions))
	for _, condition := range conditions {
		expression := condition.IR()
		if err := schema.ValidateSchemaIR(expression); err != nil {
			return Condition{}, err
		}
		if IsEmpty(expression) {
			continue
		}
		if group, ok := expression.(andExpr); ok {
			children = append(children, group.exprs...)
		} else {
			children = append(children, expression)
		}
	}
	switch len(children) {
	case 0:
		return emptyBuiltCondition(), nil
	case 1:
		return conditionFromIR(children[0])
	default:
		return conditionFromIR(andExpr{exprs: children})
	}
}

// Fingerprint identifies the exact validated predicate structure without exposing
// its contents. It is not a proof of authorization or semantic equivalence.
func (c Condition) Fingerprint() (string, error) {
	if err := ValidateCondition(c); err != nil {
		return "", err
	}
	representation, err := predicateRepresentation(c.IR())
	if err != nil {
		return "", err
	}
	data, err := json.Marshal(representation)
	if err != nil {
		return "", fmt.Errorf("%w: filter fingerprint: %w", ragy.ErrInvalidArgument, err)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

type predicateNode struct {
	Operator string          `json:"operator"`
	Field    string          `json:"field,omitempty"`
	Kind     Kind            `json:"kind,omitempty"`
	Values   []any           `json:"values,omitempty"`
	Children []predicateNode `json:"children,omitempty"`
}

func predicateRepresentation(expression IR) (predicateNode, error) {
	switch node := expression.(type) {
	case emptyExpr:
		return predicateNode{Operator: "all", Field: "", Kind: "", Values: nil, Children: nil}, nil
	case eqExpr:
		return predicateLeaf("eq", node.field, node.value), nil
	case neqExpr:
		return predicateLeaf("neq", node.field, node.value), nil
	case gtExpr:
		return predicateLeaf("gt", node.field, node.value), nil
	case gteExpr:
		return predicateLeaf("gte", node.field, node.value), nil
	case ltExpr:
		return predicateLeaf("lt", node.field, node.value), nil
	case lteExpr:
		return predicateLeaf("lte", node.field, node.value), nil
	case inExpr:
		values := make([]any, len(node.values))
		for i, value := range node.values {
			values[i] = value.Raw()
		}
		return predicateNode{
			Operator: "in",
			Field:    node.field,
			Kind:     node.values[0].Kind(),
			Values:   values,
			Children: nil,
		}, nil
	case andExpr:
		return predicateGroup("and", node.exprs)
	case orExpr:
		return predicateGroup("or", node.exprs)
	case notExpr:
		return predicateGroup("not", []IR{node.expr})
	default:
		return predicateNode{}, fmt.Errorf("%w: filter fingerprint IR", ragy.ErrInvalidArgument)
	}
}

func predicateLeaf(operator, field string, value Value) predicateNode {
	return predicateNode{
		Operator: operator,
		Field:    field,
		Kind:     value.Kind(),
		Values:   []any{value.Raw()},
		Children: nil,
	}
}

func predicateGroup(operator string, expressions []IR) (predicateNode, error) {
	children := make([]predicateNode, len(expressions))
	for i, expression := range expressions {
		child, err := predicateRepresentation(expression)
		if err != nil {
			return predicateNode{}, err
		}
		children[i] = child
	}
	return predicateNode{Operator: operator, Field: "", Kind: "", Values: nil, Children: children}, nil
}
