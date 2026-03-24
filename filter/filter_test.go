package filter

import (
	"encoding/json"
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func buildSchema(t *testing.T, builder *SchemaBuilder) Schema {
	t.Helper()

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func TestNormalizeRejectsDegenerateNodes(t *testing.T) {
	schema := NewSchema()
	field, err := schema.String("tenant")
	if err != nil {
		t.Fatalf("schema.String(): %v", err)
	}

	if _, err := Normalize(OneOf(field)); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Normalize(empty IN) error = %v", err)
	}

	if _, err := Normalize(All()); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Normalize(empty AND) error = %v", err)
	}
}

func TestNormalizeRejectsNonFiniteFloatValues(t *testing.T) {
	schema := NewSchema()
	score, err := schema.Float("score")
	if err != nil {
		t.Fatalf("schema.Float(score): %v", err)
	}

	if _, err := Normalize(Equal(score, math.NaN())); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Normalize(Equal(NaN)) error = %v, want invalid argument", err)
	}

	if _, err := Normalize(Greater(score, math.Inf(1))); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Normalize(Greater(+Inf)) error = %v, want invalid argument", err)
	}

	if _, err := Normalize(OneOf(score, math.NaN())); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Normalize(OneOf(NaN)) error = %v, want invalid argument", err)
	}
}

func TestSchemaRejectsInvalidIdentifier(t *testing.T) {
	schema := NewSchema()
	if _, err := schema.String("bad-field"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("schema.String(invalid) error = %v", err)
	}
	if _, err := schema.String("1bad"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("schema.String(leading digit) error = %v", err)
	}
	if _, err := schema.String("tenant"); err != nil {
		t.Fatalf("schema.String(valid): %v", err)
	}
	if _, err := schema.String("tenant"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("schema.String(duplicate) error = %v", err)
	}
	if _, err := schema.String("id"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("schema.String(reserved id) error = %v", err)
	}
	if _, err := schema.String("content"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("schema.String(reserved content) error = %v", err)
	}
}

func TestNormalizeBuildsTypedIR(t *testing.T) {
	schema := NewSchema()
	tenant, err := schema.String("tenant")
	if err != nil {
		t.Fatalf("schema.String(): %v", err)
	}

	status, err := schema.Int("status")
	if err != nil {
		t.Fatalf("schema.Int(): %v", err)
	}

	ir, err := Normalize(All(Equal(tenant, "acme"), Greater(status, int64(1))))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	walker := &recordingWalker{}
	if err := Walk(ir, walker); err != nil {
		t.Fatalf("Walk(): %v", err)
	}
	if walker.andCount != 1 {
		t.Fatalf("walker.andCount = %d, want 1", walker.andCount)
	}
	if len(walker.ops) != 2 || walker.ops[0] != "eq" || walker.ops[1] != "gt" {
		t.Fatalf("walker.ops = %v, want [eq gt]", walker.ops)
	}
}

func TestValidateIRRejectsCraftedInvalidNodes(t *testing.T) {
	err := ValidateIR(eqExpr{
		field: "1bad",
		value: Value{kind: KindString, raw: "x"},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateIR(invalid field) error = %v", err)
	}

	err = ValidateIR(gtExpr{
		field: "tenant",
		value: Value{kind: KindBool, raw: true},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateIR(bool gt) error = %v", err)
	}

	err = ValidateIR(inExpr{
		field: "tenant",
		values: []Value{
			{kind: KindString, raw: "acme"},
			{kind: KindInt, raw: int64(1)},
		},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateIR(mixed IN kinds) error = %v", err)
	}

	err = ValidateIR(gtExpr{
		field: "tenant",
		value: Value{kind: KindString, raw: "acme"},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateIR(string gt) error = %v", err)
	}
}

func TestSchemaValidateAttributesRejectsUndeclaredAndWrongKinds(t *testing.T) {
	builder := NewSchema()
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(): %v", err)
	}
	schema := buildSchema(t, builder)

	if err := schema.ValidateAttributes(ragy.Attributes{"tenant": "acme"}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateAttributes(undeclared) error = %v", err)
	}

	if err := schema.ValidateAttributes(ragy.Attributes{"age": "old"}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateAttributes(wrong kind) error = %v", err)
	}
}

func TestSchemaNormalizeAttributesRejectsOverflowingIntegralValues(t *testing.T) {
	builder := NewSchema()
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(age): %v", err)
	}
	schema := buildSchema(t, builder)

	if _, err := schema.NormalizeAttributes(ragy.Attributes{
		"age": float64(1e20),
	}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeAttributes(overflow float) error = %v, want invalid argument", err)
	}

	if _, err := schema.NormalizeAttributes(ragy.Attributes{
		"age": json.Number("9223372036854775808"),
	}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeAttributes(overflow json.Number) error = %v, want invalid argument", err)
	}
}

func TestSchemaValidateIRRejectsUndeclaredField(t *testing.T) {
	sourceBuilder := NewSchema()
	tenant, err := sourceBuilder.String("tenant")
	if err != nil {
		t.Fatalf("sourceBuilder.String(): %v", err)
	}
	ir, err := Normalize(Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	target := buildSchema(t, NewSchema())
	if err := target.ValidateSchemaIR(ir); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateSchemaIR(undeclared) error = %v", err)
	}
}

func TestSchemaTypedAccessorsReturnDeclaredFields(t *testing.T) {
	builder := NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(): %v", err)
	}
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(): %v", err)
	}
	if _, err := builder.Float("score"); err != nil {
		t.Fatalf("builder.Float(): %v", err)
	}
	if _, err := builder.Bool("active"); err != nil {
		t.Fatalf("builder.Bool(): %v", err)
	}
	schema := buildSchema(t, builder)

	if field, err := schema.StringField("tenant"); err != nil || field.Name() != "tenant" {
		t.Fatalf("StringField() = (%v, %v), want tenant field", field, err)
	}
	if field, err := schema.IntField("age"); err != nil || field.Name() != "age" {
		t.Fatalf("IntField() = (%v, %v), want age field", field, err)
	}
	if field, err := schema.FloatField("score"); err != nil || field.Name() != "score" {
		t.Fatalf("FloatField() = (%v, %v), want score field", field, err)
	}
	if field, err := schema.BoolField("active"); err != nil || field.Name() != "active" {
		t.Fatalf("BoolField() = (%v, %v), want active field", field, err)
	}
}

func TestSchemaTypedAccessorsRejectUndeclaredAndWrongKinds(t *testing.T) {
	builder := NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(): %v", err)
	}
	schema := buildSchema(t, builder)

	if _, err := schema.StringField("missing"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("StringField(missing) error = %v", err)
	}
	if _, err := schema.IntField("tenant"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("IntField(tenant) error = %v", err)
	}
}

func TestSchemaBuildFreezesDeclaredFields(t *testing.T) {
	builder := NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}

	schema := buildSchema(t, builder)

	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(age): %v", err)
	}

	if _, ok := schema.Lookup("age"); ok {
		t.Fatal("schema.Lookup(age) = ok, want frozen schema without post-build mutation")
	}
}

func TestResourceValidatorsRejectLeadingDigits(t *testing.T) {
	if err := ValidateSQLIdentifier("1bad"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateSQLIdentifier(): %v", err)
	}

	if err := ValidateElasticsearchIndexName("1bad"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateElasticsearchIndexName(): %v", err)
	}

	if err := ValidateCollectionName("1bad"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateCollectionName(): %v", err)
	}
}

type recordingWalker struct {
	andCount int
	ops      []string
}

func (w *recordingWalker) OnEmpty() error { w.ops = append(w.ops, "empty"); return nil }
func (w *recordingWalker) OnEq(string, Value) error {
	w.ops = append(w.ops, "eq")
	return nil
}
func (w *recordingWalker) OnNeq(string, Value) error {
	w.ops = append(w.ops, "neq")
	return nil
}
func (w *recordingWalker) OnGt(string, Value) error {
	w.ops = append(w.ops, "gt")
	return nil
}
func (w *recordingWalker) OnGte(string, Value) error {
	w.ops = append(w.ops, "gte")
	return nil
}
func (w *recordingWalker) OnLt(string, Value) error {
	w.ops = append(w.ops, "lt")
	return nil
}
func (w *recordingWalker) OnLte(string, Value) error {
	w.ops = append(w.ops, "lte")
	return nil
}
func (w *recordingWalker) OnIn(string, []Value) error {
	w.ops = append(w.ops, "in")
	return nil
}
func (w *recordingWalker) EnterAnd(int) error { w.andCount++; return nil }
func (w *recordingWalker) LeaveAnd() error    { return nil }
func (w *recordingWalker) EnterOr(int) error  { return nil }
func (w *recordingWalker) LeaveOr() error     { return nil }
func (w *recordingWalker) EnterNot() error    { return nil }
func (w *recordingWalker) LeaveNot() error    { return nil }
