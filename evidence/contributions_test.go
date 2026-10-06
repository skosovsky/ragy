package evidence_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func contributionFixture() []evidence.Contribution {
	loc := locationFixtures()[1]
	return []evidence.Contribution{
		{QueryIndex: 1, DocumentID: "first", Rank: 2, Locations: []source.Locator{loc}},
		{QueryIndex: 2, DocumentID: "second", Rank: 1, Locations: []source.Locator{loc}},
	}
}

func TestContributorExportAssociationOwnershipAndRoundtrip(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	contributions := contributionFixture()
	input.Stages[0].Hits[0].Contributions = contributions
	policy.AllowLocation = func(source.Locator) bool { return true }
	policy.AllowContribution = func(item evidence.Contribution) bool { item.Locations[0].Span.End = 999; return true }
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	hit := snapshot.Stages[0].Hits[0]
	if hit.ContributionsState != evidence.Observed || len(hit.Contributions) != 2 ||
		hit.Contributions[0].QueryIndex != 1 ||
		hit.Contributions[0].Rank != 2 ||
		*hit.Contributions[1].DocumentID.Value != "second" ||
		hit.Contributions[0].Locations[0].Span.End != 10 ||
		contributions[0].Locations[0].Span.End != 10 {
		t.Fatal("contributor association/ownership failed")
	}
	encoded, err := record.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := evidence.Decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	after, err := decoded.MarshalJSON()
	if err != nil || string(after) != string(encoded) {
		t.Fatal("contributor roundtrip failed", err)
	}
	contributions[0].Locations[0].Span.End = 7
	unchanged, err := record.MarshalJSON()
	if err != nil || string(unchanged) != string(encoded) {
		t.Fatal("record aliases contributor input", err)
	}
}

func TestContributorExportPrivacy(t *testing.T) {
	for _, denial := range []string{"default", "numbers", "identifier", "association"} {
		t.Run(denial, func(t *testing.T) {
			// Arrange.
			read, input, policy := fixture(t)
			input.Stages[0].Hits[0].Contributions = contributionFixture()[:1]
			policy.AllowContribution = func(evidence.Contribution) bool { return true }
			switch denial {
			case "default":
				policy.AllowContribution = nil
			case "numbers":
				policy.AllowNumbers = false
			case "identifier":
				policy.AllowIdentifier = func(_ evidence.IdentifierKind, value string) bool { return value != "first" }
			case "association":
				policy.AllowContribution = func(evidence.Contribution) bool { return false }
			}
			// Act.
			record, err := evidence.Capture(context.Background(), read, input, policy)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			snapshot, err := record.Snapshot()
			if err != nil || snapshot.Stages[0].Hits[0].ContributionsState != evidence.Omitted ||
				len(snapshot.Stages[0].Hits[0].Contributions) != 0 {
				t.Fatal("association bypassed export policy", err)
			}
		})
	}
}

func TestContributorValidationPrecedesPolicy(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	contribution := contributionFixture()[0]
	contribution.QueryIndex = -1
	input.Stages[0].Hits[0].Contributions = []evidence.Contribution{contribution}
	calls := 0
	policy.AllowContribution = func(evidence.Contribution) bool { calls++; return true }
	// Act.
	_, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || calls != 0 {
		t.Fatal("invalid contribution reached export callback", err)
	}
}

func TestContributorWireRejectsDuplicatesAndForeignLocations(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	input.Stages[0].Hits[0].Contributions = contributionFixture()
	policy.AllowContribution = func(evidence.Contribution) bool { return true }
	policy.AllowLocation = func(source.Locator) bool { return true }
	record, err := evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	for _, failure := range []string{"duplicate", "rank", "location"} {
		t.Run(failure, func(t *testing.T) {
			snapshot, snapshotErr := record.Snapshot()
			if snapshotErr != nil {
				t.Fatal(snapshotErr)
			}
			hit := &snapshot.Stages[0].Hits[0]
			switch failure {
			case "duplicate":
				hit.Contributions = append(hit.Contributions, hit.Contributions[0])
			case "rank":
				hit.Contributions[0].Rank = 0
			case "location":
				hit.Contributions[0].Locations[0].Span.End = 11
			}
			encoded, encodeErr := json.Marshal(snapshot)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			// Act.
			_, decodeErr := evidence.Decode(encoded)
			// Assert.
			if !errors.Is(decodeErr, ragy.ErrProtocol) {
				t.Fatal("invalid association decoded", decodeErr)
			}
		})
	}
}

func TestContributorPolicyRevocationSuppressesRecord(t *testing.T) {
	// Arrange.
	revoked := false
	read, schema := scoped(t, &revoked)
	_, input, policy := fixture(t)
	input.Schema = schema
	input.Codec = retrieval.NewJSONCodec[metadata](schema)
	input.SourceAdmission = func(context.Context, access.Binding, source.Reference) error { return nil }
	input.Stages[0].Hits[0].Contributions = contributionFixture()
	policy.AllowContribution = func(evidence.Contribution) bool { revoked = true; return true }
	// Act.
	_, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if !access.IsProtectionFailure(err) {
		t.Fatal("revocation exported query association", err)
	}
}
