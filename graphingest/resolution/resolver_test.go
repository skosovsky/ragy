package resolution_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/source"
)

type attributes struct {
	Owner string
	Tags  []string
}
type kind string
type relation string

func support(id string) source.Locator {
	return source.Locator{
		Kind: source.DocumentLocation,
		Reference: source.Reference{
			Namespace:         "n",
			Source:            id,
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          id,
			Representation:    "text",
		},
	}
}
func config() resolution.Config[kind, relation, attributes] {
	return resolution.Config[kind, relation, attributes]{
		OntologyIdentity: "service-ontology",
		PolicyIdentity:   "explicit-alias",
		MaxEntities:      20,
		MaxRelations:     20,
		MaxSupports:      60,
		ValidateEntity: func(k kind, _ attributes) error {
			if k != "Service" && k != "Database" && k != "Team" {
				return ragy.ErrInvalidGraph
			}
			return nil
		},
		ValidateRelation: func(k relation, from, to kind, _ attributes) error {
			if from != "Service" || (k != "depends_on" && k != "owned_by") || (k == "depends_on" && to != "Database") ||
				(k == "owned_by" && to != "Team") {
				return ragy.ErrInvalidGraph
			}
			return nil
		},
		Identity: func(entity resolution.Entity[kind, attributes]) (resolution.Decision, error) {
			if entity.Namespace == "" {
				return resolution.Decision{State: resolution.Ambiguous}, nil
			}
			name := entity.Name
			if entity.Namespace == "prod" && (name == "Pay" || name == "Billing") {
				name = "Billing"
			}
			return resolution.Decision{
				State:     resolution.Resolved,
				Namespace: entity.Namespace,
				Key:       string(entity.Kind) + ":" + name,
				Name:      name,
			}, nil
		},
		RelationKey:     func(edge resolution.Relation[relation, attributes]) (string, error) { return string(edge.Kind), nil },
		CloneAttributes: func(value attributes) (attributes, error) { value.Tags = slices.Clone(value.Tags); return value, nil },
		Equivalent:      func(a, b attributes) bool { return a.Owner == b.Owner && slices.Equal(a.Tags, b.Tags) },
		AdmitSupport: func(_ context.Context, _ access.Binding, location source.Locator) error {
			if location.Reference.Revision != "r1" {
				return ragy.ErrUnavailable
			}
			return nil
		},
	}
}
func entity(id, ns, name, src, owner string, k kind) resolution.Entity[kind, attributes] {
	return resolution.Entity[kind, attributes]{
		ID:         id,
		Namespace:  ns,
		Name:       name,
		Kind:       k,
		Attributes: attributes{Owner: owner, Tags: []string{"owned"}},
		Supports:   []source.Locator{support(src)},
	}
}
func fixture() resolution.Extraction[kind, relation, attributes] {
	return resolution.Extraction[kind, relation, attributes]{Entities: []resolution.Entity[kind, attributes]{
		entity(
			"a",
			"prod",
			"Pay",
			"s1",
			"Team A",
			"Service",
		),
		entity("b", "prod", "LedgerDB", "s1", "", "Database"),
		entity("c", "prod", "Team A", "s1", "", "Team"),
		entity(
			"d",
			"prod",
			"Billing",
			"s2",
			"Team A",
			"Service",
		),
		entity("e", "prod", "LedgerDB", "s2", "", "Database"),
		entity(
			"f",
			"staging",
			"Billing",
			"s3",
			"Team B",
			"Service",
		),
		entity("g", "staging", "Team B", "s3", "", "Team"),
		entity(
			"unknown",
			"",
			"Billing",
			"s4",
			"",
			"Service",
		),
		entity("conflict", "prod", "Billing", "s5", "Team B", "Service"),
	}, Relations: []resolution.Relation[relation, attributes]{
		{ID: "dep1", From: "a", To: "b", Kind: "depends_on", Supports: []source.Locator{support("s1")}},
		{ID: "owner", From: "a", To: "c", Kind: "owned_by", Supports: []source.Locator{support("s1")}},
		{ID: "dep2", From: "d", To: "e", Kind: "depends_on", Supports: []source.Locator{support("s2")}},
		{ID: "unresolved", From: "unknown", To: "b", Kind: "depends_on", Supports: []source.Locator{support("s4")}},
	}}
}
func TestNamespaceAliasAmbiguityAndConflictsRetainAllSources(t *testing.T) {
	// Arrange: explicit BYOT reference ontology and host alias policy.
	resolver, err := resolution.New(config())
	if err != nil {
		t.Fatal(err)
	}
	input := fixture()
	// Act.
	result, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
	// Assert: no namespace guessing, name-only merge or arbitrary conflict winner.
	if err != nil || len(result.Entities) != 5 || len(result.Relations) != 2 || len(result.Unresolved) != 2 {
		t.Fatal(result, err)
	}
	assertBillingGroups(t, result)
	if input.Entities[0].Attributes.Tags[0] != "owned" {
		t.Fatal("attributes aliased")
	}
	depends := 0
	for _, group := range result.Relations {
		if group.Variants[0].Kind == "depends_on" {
			depends++
			if len(group.Variants) != 1 || len(group.Variants[0].Supports) != 2 {
				t.Fatal(group)
			}
		}
	}
	if depends != 1 || result.PolicyIdentity != "explicit-alias" || result.OntologyIdentity != "service-ontology" {
		t.Fatal(result)
	}
}

func TestResolverAdmissionAndCancellationBeforePolicyCallbacks(t *testing.T) {
	for _, scenario := range []string{"limit", "malformed", "denied", "cancelled", "identity_cancel"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			cfg := config()
			input := fixture()
			callbacks := 0
			original := cfg.Identity
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg.Identity = func(entity resolution.Entity[kind, attributes]) (resolution.Decision, error) {
				callbacks++
				if scenario == "identity_cancel" {
					cancel()
				}
				return original(entity)
			}
			switch scenario {
			case "limit":
				cfg.MaxEntities = 1
			case "malformed":
				input.Entities[0].Supports[0].Reference.Artifact = ""
			case "denied":
				cfg.AdmitSupport = func(context.Context, access.Binding, source.Locator) error { return ragy.ErrUnavailable }
			case "cancelled":
				cancel()
			}
			resolver, err := resolution.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := resolver.Resolve(ctx, access.Unrestricted(), input)
			// Assert: final/protected failures never return partial identities/metadata.
			assertAdmissionFailure(t, scenario, result, err, callbacks)
		})
	}
}

func assertBillingGroups(t *testing.T, result resolution.Result[kind, relation, attributes]) {
	t.Helper()
	prod, staging := 0, 0
	for _, group := range result.Entities {
		if group.Identity.Name != "Billing" {
			continue
		}
		if group.Identity.Namespace == "prod" {
			prod++
			if len(group.Variants) != 2 || len(group.Variants[0].Supports) != 2 ||
				group.Variants[0].Attributes.Owner != "Team A" ||
				group.Variants[1].Attributes.Owner != "Team B" {
				t.Fatal(group)
			}
			group.Variants[0].Attributes.Tags[0] = "mutated"
		} else {
			staging++
			if len(group.Variants) != 1 {
				t.Fatal(group)
			}
		}
	}
	if prod != 1 || staging != 1 {
		t.Fatal("namespace or ownership failure")
	}
}

func assertAdmissionFailure(
	t *testing.T,
	scenario string,
	result resolution.Result[kind, relation, attributes],
	err error,
	callbacks int,
) {
	t.Helper()
	if err == nil || len(result.Entities) != 0 || result.PolicyIdentity != "" {
		t.Fatal(result, err)
	}
	if scenario != "identity_cancel" && callbacks != 0 {
		t.Fatal("pre-admission policy called")
	}
	if scenario == "identity_cancel" && !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestUnresolvedEndpointDoesNotBypassOntologyValidation(t *testing.T) {
	// Arrange: ambiguous namespace does not make an unsupported relation valid.
	input := fixture()
	input.Relations[3].Kind = "invented"
	resolver, err := resolution.New(config())
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidGraph) || len(result.Entities) != 0 {
		t.Fatal(result, err)
	}
}

func TestPolicyIdentityTrackedWithoutInventingEntityIdentity(t *testing.T) {
	// Arrange: same explicit decisions under two independently named policy configurations.
	firstConfig, secondConfig := config(), config()
	secondConfig.PolicyIdentity = "changed-policy"
	first, err := resolution.New(firstConfig)
	if err != nil {
		t.Fatal(err)
	}
	second, err := resolution.New(secondConfig)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	before, err := first.Resolve(context.Background(), access.Unrestricted(), fixture())
	if err != nil {
		t.Fatal(err)
	}
	after, err := second.Resolve(context.Background(), access.Unrestricted(), fixture())
	// Assert: decision provenance changes; canonical namespace/key identity remains stable.
	if err != nil || before.PolicyIdentity == after.PolicyIdentity || before.Entities[0].ID != after.Entities[0].ID {
		t.Fatal(before.PolicyIdentity, after.PolicyIdentity, err)
	}
}
