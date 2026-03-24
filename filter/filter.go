// Package filter provides a schema-bound builder and validated IR.
package filter

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/big"
	"regexp"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/ident"
)

var (
	sqlIdentifierPattern          = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	elasticsearchIndexNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	collectionNamePattern         = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]*$`)
)

const (
	jsonFloatParseBase      = 10
	jsonFloatParsePrecision = 256
)

// ValidateIdentifier enforces the shared portable field/property-name policy.
func ValidateIdentifier(name string) error {
	if !ident.IsField(name) {
		return fmt.Errorf("%w: invalid identifier %q", ragy.ErrInvalidArgument, name)
	}

	return nil
}

// ValidateSQLIdentifier enforces the unquoted-safe SQL identifier policy.
func ValidateSQLIdentifier(name string) error {
	if !sqlIdentifierPattern.MatchString(name) {
		return fmt.Errorf("%w: invalid sql identifier %q", ragy.ErrInvalidArgument, name)
	}

	return nil
}

// ValidateElasticsearchIndexName enforces the project Elasticsearch index policy.
func ValidateElasticsearchIndexName(name string) error {
	if !elasticsearchIndexNamePattern.MatchString(name) {
		return fmt.Errorf("%w: invalid elasticsearch index %q", ragy.ErrInvalidArgument, name)
	}

	return nil
}

// ValidateCollectionName enforces the project collection-name policy.
func ValidateCollectionName(name string) error {
	if !collectionNamePattern.MatchString(name) {
		return fmt.Errorf("%w: invalid collection name %q", ragy.ErrInvalidArgument, name)
	}

	return nil
}

// Kind describes a scalar field kind.
type Kind string

const (
	KindString Kind = "string"
	KindInt    Kind = "int"
	KindFloat  Kind = "float"
	KindBool   Kind = "bool"
)

type scalar interface {
	~string |
		~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~bool
}

type orderedScalar interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64
}

// SchemaBuilder constructs typed field descriptors before freezing a schema.
type SchemaBuilder struct {
	fields map[string]Kind
}

// NewSchema creates a schema builder for typed field descriptors.
func NewSchema() *SchemaBuilder {
	return &SchemaBuilder{fields: make(map[string]Kind)}
}

// Schema is a finalized immutable registry of declared fields.
type Schema struct {
	fields    map[string]Kind
	finalized bool
}

// EmptySchema constructs a finalized schema with no declared fields.
func EmptySchema() Schema {
	return Schema{
		fields:    map[string]Kind{},
		finalized: true,
	}
}

// Field is a typed field descriptor.
type Field[T scalar] struct {
	name string
	kind Kind
}

// Name returns the field name.
func (f Field[T]) Name() string {
	return f.name
}

// Kind returns the field scalar kind.
func (f Field[T]) Kind() Kind {
	return f.kind
}

func declareField[T scalar](s *SchemaBuilder, name string, kind Kind) (Field[T], error) {
	if s == nil {
		return Field[T]{}, fmt.Errorf("%w: schema", ragy.ErrInvalidArgument)
	}
	if err := ValidateIdentifier(name); err != nil {
		return Field[T]{}, err
	}
	if isReservedFieldName(name) {
		return Field[T]{}, fmt.Errorf("%w: reserved schema field %q", ragy.ErrInvalidArgument, name)
	}
	if existing, ok := s.fields[name]; ok {
		return Field[T]{}, fmt.Errorf("%w: field %q already declared as %q", ragy.ErrInvalidArgument, name, existing)
	}

	s.fields[name] = kind

	return Field[T]{name: name, kind: kind}, nil
}

// String defines a string field.
func (s *SchemaBuilder) String(name string) (Field[string], error) {
	return declareField[string](s, name, KindString)
}

// Int defines an integer field.
func (s *SchemaBuilder) Int(name string) (Field[int64], error) {
	return declareField[int64](s, name, KindInt)
}

// Float defines a float field.
func (s *SchemaBuilder) Float(name string) (Field[float64], error) {
	return declareField[float64](s, name, KindFloat)
}

// Bool defines a bool field.
func (s *SchemaBuilder) Bool(name string) (Field[bool], error) {
	return declareField[bool](s, name, KindBool)
}

// Build finalizes the builder into an immutable schema.
func (s *SchemaBuilder) Build() (Schema, error) {
	if s == nil {
		return Schema{}, fmt.Errorf("%w: schema", ragy.ErrInvalidArgument)
	}

	fields := make(map[string]Kind, len(s.fields))
	maps.Copy(fields, s.fields)

	return Schema{
		fields:    fields,
		finalized: true,
	}, nil
}

// IsFinalized reports whether the schema is safe for runtime use.
func (s Schema) IsFinalized() bool {
	return s.finalized
}

func (s Schema) validateFinalized() error {
	if !s.finalized {
		return fmt.Errorf("%w: schema", ragy.ErrInvalidArgument)
	}

	return nil
}

// Lookup returns the declared scalar kind for a field.
func (s Schema) Lookup(name string) (Kind, bool) {
	if !s.finalized {
		return "", false
	}

	kind, ok := s.fields[name]
	return kind, ok
}

func (s Schema) typedField(name string, want Kind) error {
	if err := s.validateFinalized(); err != nil {
		return err
	}

	declared, ok := s.Lookup(name)
	if !ok {
		return fmt.Errorf("%w: undeclared schema field %q", ragy.ErrInvalidArgument, name)
	}
	if declared != want {
		return fmt.Errorf(
			"%w: schema field %q declared as %q, got %q",
			ragy.ErrInvalidArgument,
			name,
			declared,
			want,
		)
	}

	return nil
}

// StringField returns a typed descriptor for a declared string field.
func (s Schema) StringField(name string) (Field[string], error) {
	if err := s.typedField(name, KindString); err != nil {
		return Field[string]{}, err
	}

	return Field[string]{name: name, kind: KindString}, nil
}

// IntField returns a typed descriptor for a declared integer field.
func (s Schema) IntField(name string) (Field[int64], error) {
	if err := s.typedField(name, KindInt); err != nil {
		return Field[int64]{}, err
	}

	return Field[int64]{name: name, kind: KindInt}, nil
}

// FloatField returns a typed descriptor for a declared float field.
func (s Schema) FloatField(name string) (Field[float64], error) {
	if err := s.typedField(name, KindFloat); err != nil {
		return Field[float64]{}, err
	}

	return Field[float64]{name: name, kind: KindFloat}, nil
}

// BoolField returns a typed descriptor for a declared bool field.
func (s Schema) BoolField(name string) (Field[bool], error) {
	if err := s.typedField(name, KindBool); err != nil {
		return Field[bool]{}, err
	}

	return Field[bool]{name: name, kind: KindBool}, nil
}

// NormalizeAttributes validates and normalizes attributes to canonical schema kinds.
func (s Schema) NormalizeAttributes(attrs ragy.Attributes) (ragy.Attributes, error) {
	if err := s.validateFinalized(); err != nil {
		return nil, err
	}
	canonical, err := ragy.NormalizeAttributes(attrs)
	if err != nil {
		return nil, err
	}
	if len(canonical) == 0 {
		var normalized ragy.Attributes
		return normalized, nil
	}

	out := make(ragy.Attributes, len(canonical))
	for name, raw := range canonical {
		kind, ok := s.Lookup(name)
		if !ok {
			return nil, fmt.Errorf("%w: undeclared schema field %q", ragy.ErrInvalidArgument, name)
		}

		normalized, err := normalizeAttributeValue(kind, raw)
		if err != nil {
			return nil, err
		}
		out[name] = normalized
	}

	return out, nil
}

// ValidateAttributes checks that attributes use only declared fields with matching kinds.
func (s Schema) ValidateAttributes(attrs ragy.Attributes) error {
	_, err := s.NormalizeAttributes(attrs)
	return err
}

// ValidateSchemaIR checks that a validated IR references only declared fields with matching kinds.
func (s Schema) ValidateSchemaIR(expr IR) error {
	if err := s.validateFinalized(); err != nil {
		return err
	}
	if err := ValidateIR(expr); err != nil {
		return err
	}

	return s.validateSchemaIR(expr)
}

// Value is a validated scalar filter value.
type Value struct {
	kind Kind
	raw  any
}

// Kind returns the scalar kind.
func (v Value) Kind() Kind {
	return v.kind
}

// Raw returns the Go value.
func (v Value) Raw() any {
	return v.raw
}

func scalarKind[T scalar](value T) Kind {
	switch any(value).(type) {
	case string:
		return KindString
	case bool:
		return KindBool
	case float32, float64:
		return KindFloat
	default:
		return KindInt
	}
}

func toValue[T scalar](value T) Value {
	switch v := any(value).(type) {
	case int:
		return Value{kind: KindInt, raw: int64(v)}
	case int8:
		return Value{kind: KindInt, raw: int64(v)}
	case int16:
		return Value{kind: KindInt, raw: int64(v)}
	case int32:
		return Value{kind: KindInt, raw: int64(v)}
	case int64:
		return Value{kind: KindInt, raw: v}
	case float32:
		return Value{kind: KindFloat, raw: float64(v)}
	case float64:
		return Value{kind: KindFloat, raw: v}
	case string:
		return Value{kind: KindString, raw: v}
	case bool:
		return Value{kind: KindBool, raw: v}
	default:
		return Value{kind: scalarKind(value), raw: value}
	}
}

// Expr is a builder node.
type Expr interface {
	isExpr()
}

type rawEq struct {
	field string
	value Value
}

func (rawEq) isExpr() {}

type rawNeq struct {
	field string
	value Value
}

func (rawNeq) isExpr() {}

type rawGt struct {
	field string
	value Value
}

func (rawGt) isExpr() {}

type rawGte struct {
	field string
	value Value
}

func (rawGte) isExpr() {}

type rawLt struct {
	field string
	value Value
}

func (rawLt) isExpr() {}

type rawLte struct {
	field string
	value Value
}

func (rawLte) isExpr() {}

type rawIn struct {
	field  string
	values []Value
}

func (rawIn) isExpr() {}

type rawAnd struct {
	exprs []Expr
}

func (rawAnd) isExpr() {}

type rawOr struct {
	exprs []Expr
}

func (rawOr) isExpr() {}

type rawNot struct {
	expr Expr
}

func (rawNot) isExpr() {}

// Equal creates an equality predicate.
func Equal[T scalar](field Field[T], value T) Expr {
	return rawEq{field: field.name, value: toValue(value)}
}

// NotEqual creates a not-equal predicate.
func NotEqual[T scalar](field Field[T], value T) Expr {
	return rawNeq{field: field.name, value: toValue(value)}
}

// Greater creates a greater-than predicate.
func Greater[T orderedScalar](field Field[T], value T) Expr {
	return rawGt{field: field.name, value: toValue(value)}
}

// GreaterOrEqual creates a greater-or-equal predicate.
func GreaterOrEqual[T orderedScalar](field Field[T], value T) Expr {
	return rawGte{field: field.name, value: toValue(value)}
}

// Less creates a less-than predicate.
func Less[T orderedScalar](field Field[T], value T) Expr {
	return rawLt{field: field.name, value: toValue(value)}
}

// LessOrEqual creates a less-or-equal predicate.
func LessOrEqual[T orderedScalar](field Field[T], value T) Expr {
	return rawLte{field: field.name, value: toValue(value)}
}

// OneOf creates a membership predicate.
func OneOf[T scalar](field Field[T], values ...T) Expr {
	normalized := make([]Value, 0, len(values))
	for _, value := range values {
		normalized = append(normalized, toValue(value))
	}

	return rawIn{field: field.name, values: normalized}
}

// All creates a logical AND predicate.
func All(exprs ...Expr) Expr {
	return rawAnd{exprs: exprs}
}

// Any creates a logical OR predicate.
func Any(exprs ...Expr) Expr {
	return rawOr{exprs: exprs}
}

// Inverse creates a logical NOT predicate.
func Inverse(expr Expr) Expr {
	return rawNot{expr: expr}
}

// IR is the validated filter representation consumed by adapters.
type IR interface {
	isIR()
}

type emptyExpr struct{}

func (emptyExpr) isIR() {}

type eqExpr struct {
	field string
	value Value
}

func (eqExpr) isIR() {}

type neqExpr struct {
	field string
	value Value
}

func (neqExpr) isIR() {}

type gtExpr struct {
	field string
	value Value
}

func (gtExpr) isIR() {}

type gteExpr struct {
	field string
	value Value
}

func (gteExpr) isIR() {}

type ltExpr struct {
	field string
	value Value
}

func (ltExpr) isIR() {}

type lteExpr struct {
	field string
	value Value
}

func (lteExpr) isIR() {}

type inExpr struct {
	field  string
	values []Value
}

func (inExpr) isIR() {}

type andExpr struct {
	exprs []IR
}

func (andExpr) isIR() {}

type orExpr struct {
	exprs []IR
}

func (orExpr) isIR() {}

type notExpr struct {
	expr IR
}

func (notExpr) isIR() {}

// Walker consumes a validated filter tree.
type Walker interface {
	OnEmpty() error
	OnEq(field string, value Value) error
	OnNeq(field string, value Value) error
	OnGt(field string, value Value) error
	OnGte(field string, value Value) error
	OnLt(field string, value Value) error
	OnLte(field string, value Value) error
	OnIn(field string, values []Value) error
	EnterAnd(n int) error
	LeaveAnd() error
	EnterOr(n int) error
	LeaveOr() error
	EnterNot() error
	LeaveNot() error
}

// IsEmpty reports whether expr represents the absence of a filter.
func IsEmpty(expr IR) bool {
	switch expr.(type) {
	case nil, emptyExpr:
		return true
	default:
		return false
	}
}

// Walk traverses a validated filter tree.
func Walk(expr IR, walker Walker) error {
	switch node := expr.(type) {
	case nil, emptyExpr:
		return walker.OnEmpty()
	case eqExpr, neqExpr, gtExpr, gteExpr, ltExpr, lteExpr, inExpr:
		return walkLeaf(node, walker)
	case andExpr:
		return walkGroup(node.exprs, walker.EnterAnd, walker.LeaveAnd, walker)
	case orExpr:
		return walkGroup(node.exprs, walker.EnterOr, walker.LeaveOr, walker)
	case notExpr:
		if err := walker.EnterNot(); err != nil {
			return err
		}
		if err := Walk(node.expr, walker); err != nil {
			return err
		}
		return walker.LeaveNot()
	default:
		return fmt.Errorf("%w: unknown filter IR %T", ragy.ErrInvalidArgument, expr)
	}
}

func walkLeaf(node IR, walker Walker) error {
	switch leaf := node.(type) {
	case eqExpr:
		return walker.OnEq(leaf.field, leaf.value)
	case neqExpr:
		return walker.OnNeq(leaf.field, leaf.value)
	case gtExpr:
		return walker.OnGt(leaf.field, leaf.value)
	case gteExpr:
		return walker.OnGte(leaf.field, leaf.value)
	case ltExpr:
		return walker.OnLt(leaf.field, leaf.value)
	case lteExpr:
		return walker.OnLte(leaf.field, leaf.value)
	case inExpr:
		return walker.OnIn(leaf.field, append([]Value(nil), leaf.values...))
	default:
		return fmt.Errorf("%w: unknown filter IR %T", ragy.ErrInvalidArgument, node)
	}
}

func walkGroup(
	items []IR,
	enter func(int) error,
	leave func() error,
	walker Walker,
) error {
	if err := enter(len(items)); err != nil {
		return err
	}
	for _, child := range items {
		if err := Walk(child, walker); err != nil {
			return err
		}
	}
	return leave()
}

// Normalize validates and normalizes a builder expression to IR.
func Normalize(expr Expr) (IR, error) {
	switch e := expr.(type) {
	case nil:
		return emptyExpr{}, nil
	case rawEq:
		return eqExpr(e), validateComparison(e.field, e.value)
	case rawNeq:
		return neqExpr(e), validateComparison(e.field, e.value)
	case rawGt:
		return gtExpr(e), validateComparison(e.field, e.value)
	case rawGte:
		return gteExpr(e), validateComparison(e.field, e.value)
	case rawLt:
		return ltExpr(e), validateComparison(e.field, e.value)
	case rawLte:
		return lteExpr(e), validateComparison(e.field, e.value)
	case rawIn:
		if err := ValidateIdentifier(e.field); err != nil {
			return nil, err
		}

		if len(e.values) == 0 {
			return nil, fmt.Errorf("%w: empty IN predicate", ragy.ErrInvalidArgument)
		}

		values := append([]Value(nil), e.values...)
		expr := inExpr{field: e.field, values: values}

		return expr, validateIn(expr)
	case rawAnd:
		if len(e.exprs) == 0 {
			return nil, fmt.Errorf("%w: empty AND predicate", ragy.ErrInvalidArgument)
		}

		return normalizeGroup(e.exprs, true)
	case rawOr:
		if len(e.exprs) == 0 {
			return nil, fmt.Errorf("%w: empty OR predicate", ragy.ErrInvalidArgument)
		}

		return normalizeGroup(e.exprs, false)
	case rawNot:
		if e.expr == nil {
			return nil, fmt.Errorf("%w: nil NOT predicate", ragy.ErrInvalidArgument)
		}

		normalized, err := Normalize(e.expr)
		if err != nil {
			return nil, err
		}

		return notExpr{expr: normalized}, nil
	default:
		return nil, fmt.Errorf("%w: unknown filter node %T", ragy.ErrInvalidArgument, expr)
	}
}

// ValidateIR validates a filter IR tree assembled at runtime.
func ValidateIR(expr IR) error {
	switch node := expr.(type) {
	case nil:
		return nil
	case emptyExpr:
		return nil
	case eqExpr:
		return validateComparison(node.field, node.value)
	case neqExpr:
		return validateComparison(node.field, node.value)
	case gtExpr:
		return validateOrderedComparison(node.field, node.value)
	case gteExpr:
		return validateOrderedComparison(node.field, node.value)
	case ltExpr:
		return validateOrderedComparison(node.field, node.value)
	case lteExpr:
		return validateOrderedComparison(node.field, node.value)
	case inExpr:
		return validateIn(node)
	case andExpr:
		return validateGroup("AND", node.exprs)
	case orExpr:
		return validateGroup("OR", node.exprs)
	case notExpr:
		return validateNot(node)
	default:
		return fmt.Errorf("%w: unknown filter IR %T", ragy.ErrInvalidArgument, expr)
	}
}

func validateIn(node inExpr) error {
	if err := ValidateIdentifier(node.field); err != nil {
		return err
	}

	if len(node.values) == 0 {
		return fmt.Errorf("%w: empty IN predicate", ragy.ErrInvalidArgument)
	}

	firstKind := node.values[0].Kind()
	for _, value := range node.values {
		if err := validateValue(value); err != nil {
			return err
		}
		if value.Kind() != firstKind {
			return fmt.Errorf("%w: mixed IN predicate value kinds", ragy.ErrInvalidArgument)
		}
	}

	return nil
}

func validateGroup(name string, exprs []IR) error {
	if len(exprs) == 0 {
		return fmt.Errorf("%w: empty %s predicate", ragy.ErrInvalidArgument, name)
	}

	for _, expr := range exprs {
		if expr == nil {
			return fmt.Errorf("%w: nil group member", ragy.ErrInvalidArgument)
		}
		if err := ValidateIR(expr); err != nil {
			return err
		}
	}

	return nil
}

func validateNot(node notExpr) error {
	if node.expr == nil {
		return fmt.Errorf("%w: nil NOT predicate", ragy.ErrInvalidArgument)
	}
	return ValidateIR(node.expr)
}

func normalizeGroup(exprs []Expr, isAnd bool) (IR, error) {
	items := make([]IR, 0, len(exprs))
	for _, expr := range exprs {
		if expr == nil {
			return nil, fmt.Errorf("%w: nil group member", ragy.ErrInvalidArgument)
		}

		normalized, err := Normalize(expr)
		if err != nil {
			return nil, err
		}

		switch node := normalized.(type) {
		case andExpr:
			if isAnd {
				items = append(items, node.exprs...)
				continue
			}
		case orExpr:
			if !isAnd {
				items = append(items, node.exprs...)
				continue
			}
		}

		items = append(items, normalized)
	}

	if isAnd {
		return andExpr{exprs: items}, nil
	}

	return orExpr{exprs: items}, nil
}

func validateComparison(field string, value Value) error {
	if err := ValidateIdentifier(field); err != nil {
		return err
	}

	return validateValue(value)
}

func validateOrderedComparison(field string, value Value) error {
	if err := validateComparison(field, value); err != nil {
		return err
	}

	switch value.Kind() {
	case KindInt, KindFloat:
		return nil
	case KindString, KindBool:
		return fmt.Errorf("%w: ordered comparison kind %q", ragy.ErrInvalidArgument, value.Kind())
	default:
		return fmt.Errorf("%w: ordered comparison kind %q", ragy.ErrInvalidArgument, value.Kind())
	}
}

func validateValue(value Value) error {
	if value.Raw() == nil {
		return fmt.Errorf("%w: nil comparison value", ragy.ErrInvalidArgument)
	}

	switch value.Kind() {
	case KindString:
		if _, ok := value.Raw().(string); !ok {
			return fmt.Errorf("%w: string value must use string raw type", ragy.ErrInvalidArgument)
		}
	case KindInt:
		if _, ok := value.Raw().(int64); !ok {
			return fmt.Errorf("%w: int value must use int64 raw type", ragy.ErrInvalidArgument)
		}
	case KindFloat:
		floatValue, ok := value.Raw().(float64)
		if !ok {
			return fmt.Errorf("%w: float value must use float64 raw type", ragy.ErrInvalidArgument)
		}
		if err := validateFiniteFloat(floatValue); err != nil {
			return err
		}
	case KindBool:
		if _, ok := value.Raw().(bool); !ok {
			return fmt.Errorf("%w: bool value must use bool raw type", ragy.ErrInvalidArgument)
		}
	default:
		return fmt.Errorf("%w: unknown value kind %q", ragy.ErrInvalidArgument, value.Kind())
	}

	return nil
}

func normalizeAttributeValue(kind Kind, raw any) (any, error) {
	switch kind {
	case KindString:
		value, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("%w: schema field requires string value", ragy.ErrInvalidArgument)
		}
		return value, nil
	case KindInt:
		return normalizeIntAttribute(raw)
	case KindFloat:
		return normalizeFloatAttribute(raw)
	case KindBool:
		value, ok := raw.(bool)
		if !ok {
			return nil, fmt.Errorf("%w: schema field requires bool value", ragy.ErrInvalidArgument)
		}
		return value, nil
	default:
		return nil, fmt.Errorf("%w: unknown schema field kind %q", ragy.ErrInvalidArgument, kind)
	}
}

func normalizeIntAttribute(raw any) (int64, error) {
	switch value := raw.(type) {
	case int:
		return int64(value), nil
	case int8:
		return int64(value), nil
	case int16:
		return int64(value), nil
	case int32:
		return int64(value), nil
	case int64:
		return value, nil
	case uint, uint8, uint16, uint32, uint64:
		return 0, fmt.Errorf("%w: schema field requires int64 value", ragy.ErrInvalidArgument)
	case float32:
		return normalizeIntegralFloatToInt64(float64(value))
	case float64:
		return normalizeIntegralFloatToInt64(value)
	case json.Number:
		if parsed, err := value.Int64(); err == nil {
			return parsed, nil
		}
		return normalizeIntegralJSONNumberToInt64(value)
	default:
		return 0, fmt.Errorf("%w: schema field requires int64 value", ragy.ErrInvalidArgument)
	}
}

func validateFiniteFloat(value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("%w: float value must be finite", ragy.ErrInvalidArgument)
	}

	return nil
}

func normalizeIntegralFloatToInt64(value float64) (int64, error) {
	if err := validateFiniteFloat(value); err != nil {
		return 0, fmt.Errorf("%w: schema field requires integer value", ragy.ErrInvalidArgument)
	}

	parsed, accuracy := big.NewFloat(value).Int64()
	if accuracy != big.Exact {
		return 0, fmt.Errorf("%w: schema field requires integer value", ragy.ErrInvalidArgument)
	}

	return parsed, nil
}

func normalizeIntegralJSONNumberToInt64(value json.Number) (int64, error) {
	parsed, _, err := big.ParseFloat(
		value.String(),
		jsonFloatParseBase,
		jsonFloatParsePrecision,
		big.ToZero,
	)
	if err != nil {
		return 0, fmt.Errorf("%w: schema field requires integer value", ragy.ErrInvalidArgument)
	}

	out, accuracy := parsed.Int64()
	if accuracy != big.Exact {
		return 0, fmt.Errorf("%w: schema field requires integer value", ragy.ErrInvalidArgument)
	}

	return out, nil
}

func normalizeFloatAttribute(raw any) (float64, error) {
	switch value := raw.(type) {
	case int:
		return float64(value), nil
	case int8:
		return float64(value), nil
	case int16:
		return float64(value), nil
	case int32:
		return float64(value), nil
	case int64:
		return float64(value), nil
	case float32:
		return float64(value), nil
	case float64:
		return value, nil
	case json.Number:
		parsed, err := value.Float64()
		if err != nil {
			return 0, fmt.Errorf("%w: schema field requires float64 value", ragy.ErrInvalidArgument)
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("%w: schema field requires float64 value", ragy.ErrInvalidArgument)
	}
}

func (s Schema) validateSchemaIR(expr IR) error {
	switch node := expr.(type) {
	case nil, emptyExpr:
		return nil
	case eqExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case neqExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case gtExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case gteExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case ltExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case lteExpr:
		return s.validateFieldKind(node.field, node.value.Kind())
	case inExpr:
		return s.validateInKinds(node)
	case andExpr:
		return s.validateSchemaGroup(node.exprs)
	case orExpr:
		return s.validateSchemaGroup(node.exprs)
	case notExpr:
		return s.validateSchemaIR(node.expr)
	default:
		return fmt.Errorf("%w: unknown filter IR %T", ragy.ErrInvalidArgument, expr)
	}
}

func (s Schema) validateSchemaGroup(exprs []IR) error {
	for _, expr := range exprs {
		if err := s.validateSchemaIR(expr); err != nil {
			return err
		}
	}

	return nil
}

func (s Schema) validateInKinds(node inExpr) error {
	if len(node.values) == 0 {
		return fmt.Errorf("%w: empty IN predicate", ragy.ErrInvalidArgument)
	}

	return s.validateFieldKind(node.field, node.values[0].Kind())
}

func (s Schema) validateFieldKind(field string, got Kind) error {
	declared, ok := s.Lookup(field)
	if !ok {
		return fmt.Errorf("%w: undeclared schema field %q", ragy.ErrInvalidArgument, field)
	}
	if declared != got {
		return fmt.Errorf("%w: schema field %q declared as %q, got %q", ragy.ErrInvalidArgument, field, declared, got)
	}

	return nil
}

func isReservedFieldName(name string) bool {
	switch name {
	case "id", "content", "type", "source_id", "target_id":
		return true
	default:
		return false
	}
}
