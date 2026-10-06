package filter

import (
	"errors"
	"sync"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestIntersectPreservesMandatoryConstraint(t *testing.T) {
	t.Parallel()
	// Arrange.
	builder := NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	first, _ := NewBuilder(schema)
	second, _ := NewBuilder(schema)
	mandatory, err := Eq(first, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	planned, err := Eq(second, tenant, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	effective, intersectErr := Intersect(schema, mandatory, planned)
	// Assert: conflicting scope is not converted to no filter.
	if intersectErr != nil || IsEmpty(effective.IR()) {
		t.Fatalf("intersect=%v error=%v", effective, intersectErr)
	}
	for _, value := range []string{"a", "b", "c"} {
		matched, matchErr := MatchCondition(effective, func(string) (any, bool) { return value, true })
		if matchErr != nil || matched {
			t.Fatalf("value=%q matched=%v error=%v", value, matched, matchErr)
		}
	}
	// Act: a removed query filter does not remove the mandatory scope.
	effective, intersectErr = Intersect(schema, mandatory, Condition{})
	matched, matchErr := MatchCondition(effective, func(string) (any, bool) { return "b", true })
	// Assert.
	if intersectErr != nil || matchErr != nil || matched {
		t.Fatalf("intersection=%v match=%v allowed=%v", intersectErr, matchErr, matched)
	}
}

func TestIntersectRejectsIncompatibleSchema(t *testing.T) {
	t.Parallel()
	// Arrange.
	source := NewSchema()
	field, err := source.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	sourceSchema, err := source.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, _ := NewBuilder(sourceSchema)
	condition, err := Eq(builder, field, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	target := NewSchema()
	if _, declareErr := target.Int("tenant"); declareErr != nil {
		t.Fatal(declareErr)
	}
	targetSchema, err := target.Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, wrongType := Intersect(targetSchema, condition)
	_, missingField := Intersect(EmptySchema(), condition)
	_, unfinalized := Intersect(Schema{}, condition)
	// Assert.
	for _, failure := range []error{wrongType, missingField, unfinalized} {
		if !errors.Is(failure, ragy.ErrInvalidArgument) {
			t.Fatalf("error=%v", failure)
		}
	}
}

func TestConditionFingerprintStableAcrossConcurrentComposition(t *testing.T) {
	t.Parallel()
	// Arrange.
	schemaBuilder := NewSchema()
	tenant, err := schemaBuilder.Int("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := schemaBuilder.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, _ := NewBuilder(schema)
	values := []int64{9007199254740992, 9007199254740993}
	original, err := In(builder, tenant, values...).Build()
	if err != nil {
		t.Fatal(err)
	}
	digest, err := original.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	values[0] = 0
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			combined, composeErr := Intersect(schema, Condition{}, original)
			got, fingerprintErr := combined.Fingerprint()
			// Assert.
			if composeErr != nil || fingerprintErr != nil || got != digest {
				t.Errorf("compose=%v fingerprint=%v got=%q want=%q", composeErr, fingerprintErr, got, digest)
			}
		})
	}
	workers.Wait()
	// Act and Assert: membership value changes produce a different identity.
	otherBuilder, _ := NewBuilder(schema)
	changed, err := In(otherBuilder, tenant, int64(0)).Build()
	if err != nil {
		t.Fatal(err)
	}
	otherDigest, err := changed.Fingerprint()
	if err != nil || otherDigest == digest {
		t.Fatalf("other=%q original=%q error=%v", otherDigest, digest, err)
	}
}
