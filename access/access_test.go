package access_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
)

func TestBindingScopeAndFreshness(t *testing.T) {
	// Arrange.
	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	first, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(first, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	second, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	conflict, err := filter.Eq(second, tenant, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	issued := now
	revoked := false
	calls := 0
	binding, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "host-policy-7",
			PolicyEpoch: 7,
			IssuedAt:    issued,
			ExpiresAt:   issued.Add(30 * time.Second),
		},
		Schema:      schema,
		Mandatory:   mandatory,
		Publication: access.CurrentPublication(),
		Now:         func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			calls++
			if revoked {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Act: contradictory query keeps a restrictive intersection.
	effective, err := binding.Prepare(
		context.Background(),
		schema,
		conflict,
		access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"a", "b"} {
		matches, matchErr := filter.MatchCondition(
			effective,
			func(field string) (any, bool) { return value, field == "tenant" },
		)
		if matchErr != nil || matches {
			t.Fatalf("contradiction expanded access: %s, %v, %v", value, matches, matchErr)
		}
	}
	// Act: revoke inside TTL.
	now = issued.Add(10 * time.Second)
	revoked = true
	err = binding.Check(context.Background())
	// Assert.
	if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("revocation accepted: %v", err)
	}
	// Act: expiry requires a new host decision/binding; the old snapshot cannot extend itself.
	revoked = false
	now = issued.Add(31 * time.Second)
	before := calls
	err = binding.Check(context.Background())
	// Assert.
	if !access.IsProtectionFailure(err) || calls != before {
		t.Fatalf("expired authorization reused: %v", err)
	}
}

func TestBindingAndPublicationAreImmutable(t *testing.T) {
	// Arrange.
	inventory := []access.TargetRevision{
		{Target: "tensor", Namespace: "n", Source: "s", Revision: "r1", Transformation: "t1", AccessFingerprint: "a1"},
	}
	publication, err := access.PinPublication("pub1", inventory)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	inventory[0].Revision = "r2"
	exposed := binding.Publication().Targets()
	exposed[0].Revision = "r3"
	// Assert.
	if binding.Publication().Targets()[0].Revision != "r1" {
		t.Fatal("publication snapshot mutated")
	}
	if err := (access.Binding{}).Check(context.Background()); !access.IsProtectionFailure(err) {
		t.Fatalf("zero binding accepted: %v", err)
	}
	if err := access.Unrestricted().Check(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestMissingCapabilityRejectsBeforeTargetDispatch(t *testing.T) {
	// Arrange.
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatal(err)
	}
	publication, err := access.PinPublication(
		"pub1",
		[]access.TargetRevision{
			{
				Target:            "dense",
				Namespace:         "n",
				Source:            "s",
				Revision:          "r1",
				Transformation:    "t1",
				AccessFingerprint: "a1",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = binding.Prepare(
		context.Background(),
		schema,
		filter.Condition{},
		access.Capabilities{RequirePinnedPublication: false},
	)
	// Assert.
	if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatalf("unsupported pin admitted: %v", err)
	}
}

func TestProtectionErrorDoesNotExposePolicyValues(t *testing.T) {
	// Arrange.
	protected := access.Protect(errors.New("private-object-id"))
	// Act.
	text := protected.Error()
	// Assert.
	if strings.Contains(text, "private-object-id") {
		t.Fatal("protection diagnostics leak identifiers")
	}
}

func TestProtectionClassificationDropsJoinedSideErrors(t *testing.T) {
	// Arrange: sibling errors can contain arbitrary backend details or typed payloads.
	protected := access.Protect(ragy.ErrUnsupported)
	sideError := errors.New("private-object-id")
	joined := errors.Join(protected, sideError)
	// Act.
	collapsed := access.Protect(joined)
	// Assert: keep the protection classification and discard unrelated joined data.
	if errors.Is(collapsed, sideError) || !errors.Is(collapsed, ragy.ErrUnsupported) ||
		strings.Contains(collapsed.Error(), "private-object-id") {
		t.Fatalf("joined side error escaped protection boundary: %v", collapsed)
	}
}
