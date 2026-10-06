package resolution_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/source"
)

func TestMalformedInputIdentityRejectedBeforeCallbacks(t *testing.T) {
	setters := map[string]func(*resolution.Extraction[kind, relation, attributes], string){
		"entity_id":   func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Entities[8].ID = v },
		"namespace":   func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Entities[8].Namespace = v },
		"name":        func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Entities[8].Name = v },
		"relation_id": func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Relations[3].ID = v },
		"from":        func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Relations[3].From = v },
		"to":          func(x *resolution.Extraction[kind, relation, attributes], v string) { x.Relations[3].To = v },
	}
	for field, set := range setters {
		for _, bad := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
			t.Run(fmt.Sprintf("%s/%x", field, bad), func(t *testing.T) {
				// Arrange: malformed field late in batch; count all payload callbacks.
				cfg, input := config(), fixture()
				callbacks := 0
				cfg.CloneAttributes = func(a attributes) (attributes, error) { callbacks++; return a, nil }
				cfg.AdmitSupport = func(context.Context, access.Binding, source.Locator) error { callbacks++; return nil }
				original := cfg.Identity
				cfg.Identity = func(e resolution.Entity[kind, attributes]) (resolution.Decision, error) {
					callbacks++
					return original(e)
				}
				set(&input, bad)
				resolver, err := resolution.New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				// Act.
				result, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
				// Assert: whole structural batch precedes admission, clone and identity.
				if !errors.Is(err, ragy.ErrInvalidArgument) ||
					!reflect.DeepEqual(result, resolution.Result[kind, relation, attributes]{}) ||
					callbacks != 0 {
					t.Fatal(result, err, callbacks)
				}
			})
		}
	}
}

func TestMalformedConfigurationIdentity(t *testing.T) {
	for _, field := range []string{"ontology", "policy"} {
		for _, bad := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
			t.Run(fmt.Sprintf("%s/%x", field, bad), func(t *testing.T) {
				// Arrange.
				cfg := config()
				if field == "ontology" {
					cfg.OntologyIdentity = bad
				} else {
					cfg.PolicyIdentity = bad
				}
				// Act.
				resolver, err := resolution.New(cfg)
				// Assert.
				if resolver != nil || !errors.Is(err, ragy.ErrInvalidArgument) {
					t.Fatal(resolver, err)
				}
			})
		}
	}
}

func TestMalformedPolicyIdentitiesSuppressAllOutput(t *testing.T) {
	for _, field := range []string{"namespace", "key", "name", "relation_key"} {
		for _, bad := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
			t.Run(fmt.Sprintf("%s/%x", field, bad), func(t *testing.T) {
				// Arrange: inject first malformed policy outcome, stop before next policy or grouping.
				cfg := config()
				identities, keys, equivalents := 0, 0, 0
				original := cfg.Identity
				cfg.Identity = func(e resolution.Entity[kind, attributes]) (resolution.Decision, error) {
					identities++
					d, err := original(e)
					if identities == 1 {
						corruptDecision(&d, field, bad)
					}
					return d, err
				}
				cfg.RelationKey = func(resolution.Relation[relation, attributes]) (string, error) { keys++; return bad, nil }
				cfg.Equivalent = func(attributes, attributes) bool { equivalents++; return true }
				resolver, err := resolution.New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				// Act.
				result, err := resolver.Resolve(context.Background(), access.Unrestricted(), fixture())
				// Assert: no partial result, no next identity/group-equivalence after bad decision.
				assertPolicyFailure(t, field, result, err, identities, keys, equivalents)
			})
		}
	}
}

func corruptDecision(d *resolution.Decision, field, value string) {
	switch field {
	case "namespace":
		d.Namespace = value
	case "key":
		d.Key = value
	case "name":
		d.Name = value
	}
}

func TestValidUnicodeTuplesKeepLegacyIDsAndDistinctBoundaries(t *testing.T) {
	// Arrange: valid replacement character, embedded punctuation, tuple boundaries,
	// canonically equivalent Unicode and different case remain host-distinct.
	tuples := [][2]string{
		{"�", "�"},
		{"a", "bc"},
		{"ab", "c"},
		{"é", "x"},
		{"e\u0301", "x"},
		{"A", "bc"},
		{"a\x00", "bc"},
		{"a", "bc"},
	}
	input := resolution.Extraction[kind, relation, attributes]{}
	decisions := make(map[string]resolution.Decision)
	for i, tuple := range tuples {
		id := strconv.Itoa(i)
		input.Entities = append(input.Entities, entity(id, tuple[0], "�", "s1", "", "Service"))
		decisions[id] = resolution.Decision{State: resolution.Resolved, Namespace: tuple[0], Key: tuple[1], Name: "�"}
	}
	cfg := config()
	cfg.OntologyIdentity, cfg.PolicyIdentity = "�", "�"
	cfg.Identity = func(e resolution.Entity[kind, attributes]) (resolution.Decision, error) { return decisions[e.ID], nil }
	cfg.ValidateRelation = func(relation, kind, kind, attributes) error { return nil }
	cfg.RelationKey = func(e resolution.Relation[relation, attributes]) (string, error) { return string(e.Kind), nil }
	for i, key := range []string{"�", "distinct", "�"} {
		input.Relations = append(
			input.Relations,
			resolution.Relation[relation, attributes]{
				ID:       strconv.Itoa(i),
				From:     "0",
				To:       "1",
				Kind:     relation(key),
				Supports: []source.Locator{support("s1")},
			},
		)
	}
	resolver, err := resolution.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
	// Assert: equal tuples merge, every distinct tuple retains its historical hash.
	if err != nil || len(result.Entities) != len(tuples)-1 || len(result.Relations) != 2 {
		t.Fatal(result, err)
	}
	for i, trace := range result.EntityDecisions {
		if trace.CanonicalID != legacyID(t, "entity", tuples[i][0], tuples[i][1]) {
			t.Fatal(trace)
		}
	}
	for i, trace := range result.RelationDecisions {
		if trace.CanonicalID != legacyID(
			t,
			"relation",
			result.EntityDecisions[0].CanonicalID,
			result.EntityDecisions[1].CanonicalID,
			string(input.Relations[i].Kind),
		) {
			t.Fatal(trace)
		}
	}
}

func legacyID(t *testing.T, kind string, parts ...string) string {
	t.Helper()
	data, err := json.Marshal(parts)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return kind + ":" + hex.EncodeToString(digest[:])
}

func assertPolicyFailure(
	t *testing.T,
	field string,
	result resolution.Result[kind, relation, attributes],
	err error,
	identities, keys, equivalents int,
) {
	t.Helper()
	if !errors.Is(err, ragy.ErrProtocol) ||
		!reflect.DeepEqual(result, resolution.Result[kind, relation, attributes]{}) {
		t.Fatal(result, err)
	}
	if field == "relation_key" {
		if keys != 1 {
			t.Fatal("later relation policy", keys)
		}
	} else if identities != 1 || keys != 0 || equivalents != 0 {
		t.Fatal(identities, keys, equivalents)
	}
}
