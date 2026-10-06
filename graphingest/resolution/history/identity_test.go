//go:build darwin || linux

package history_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution/history"
	"github.com/skosovsky/ragy/source"
)

func TestCaptureRejectsMalformedIdentityBeforeSerialization(t *testing.T) {
	for field, set := range identitySetters() {
		for _, bad := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
			t.Run(fmt.Sprintf("%s/%x", field, bad), func(t *testing.T) {
				// Arrange: valid resolver record, then malformed declared identity.
				input := record(t, "r1", "TeamA", "policy", "")
				set(&input, bad)
				admissions := 0
				admit := func(context.Context, access.Binding, source.Locator) error { admissions++; return nil }
				// Act.
				snapshot, err := history.Capture(
					context.Background(),
					access.Unrestricted(),
					input,
					admit,
					maxBytes,
					maxSupports,
				)
				// Assert: all malformed identities precede admission/serialization; no repair.
				if err == nil || (!errors.Is(err, ragy.ErrInvalidArgument) && !errors.Is(err, ragy.ErrProtocol)) ||
					!reflect.DeepEqual(snapshot.Reference(), history.Reference{}) || admissions != 0 {
					t.Fatal(snapshot.Reference(), err, admissions)
				}
			})
		}
	}
}

func identitySetters() map[string]func(*history.Record[string, string, attributes], string) {
	return map[string]func(*history.Record[string, string, attributes], string){
		"run":                      func(r *history.Record[string, string, attributes], s string) { r.Metadata.Run = s },
		"extraction_configuration": func(r *history.Record[string, string, attributes], s string) { r.Metadata.ExtractionFingerprint = s },
		"parent":                   func(r *history.Record[string, string, attributes], s string) { r.Metadata.Parent = s },
		"input_entity_id":          func(r *history.Record[string, string, attributes], s string) { r.Input.Entities[0].ID = s },
		"input_namespace":          func(r *history.Record[string, string, attributes], s string) { r.Input.Entities[0].Namespace = s },
		"input_name":               func(r *history.Record[string, string, attributes], s string) { r.Input.Entities[0].Name = s },
		"input_relation_id":        func(r *history.Record[string, string, attributes], s string) { r.Input.Relations[0].ID = s },
		"input_from":               func(r *history.Record[string, string, attributes], s string) { r.Input.Relations[0].From = s },
		"input_to":                 func(r *history.Record[string, string, attributes], s string) { r.Input.Relations[0].To = s },
		"ontology":                 func(r *history.Record[string, string, attributes], s string) { r.Result.OntologyIdentity = s },
		"policy":                   func(r *history.Record[string, string, attributes], s string) { r.Result.PolicyIdentity = s },
		"group_entity_id":          func(r *history.Record[string, string, attributes], s string) { r.Result.Entities[0].ID = s },
		"group_namespace": func(r *history.Record[string, string, attributes], s string) {
			r.Result.Entities[0].Identity.Namespace = s
		},
		"group_key":          func(r *history.Record[string, string, attributes], s string) { r.Result.Entities[0].Identity.Key = s },
		"group_name":         func(r *history.Record[string, string, attributes], s string) { r.Result.Entities[0].Identity.Name = s },
		"group_relation_id":  func(r *history.Record[string, string, attributes], s string) { r.Result.Relations[0].ID = s },
		"group_from":         func(r *history.Record[string, string, attributes], s string) { r.Result.Relations[0].From = s },
		"group_to":           func(r *history.Record[string, string, attributes], s string) { r.Result.Relations[0].To = s },
		"unresolved_mention": func(r *history.Record[string, string, attributes], s string) { r.Result.Unresolved[0].Mention = s },
		"unresolved_kind":    func(r *history.Record[string, string, attributes], s string) { r.Result.Unresolved[0].Kind = s },
		"entity_mention":     func(r *history.Record[string, string, attributes], s string) { r.Result.EntityDecisions[0].Mention = s },
		"entity_namespace": func(r *history.Record[string, string, attributes], s string) {
			r.Result.EntityDecisions[0].Identity.Namespace = s
		},
		"entity_key": func(r *history.Record[string, string, attributes], s string) {
			r.Result.EntityDecisions[0].Identity.Key = s
		},
		"entity_name": func(r *history.Record[string, string, attributes], s string) {
			r.Result.EntityDecisions[0].Identity.Name = s
		},
		"entity_canonical": func(r *history.Record[string, string, attributes], s string) {
			r.Result.EntityDecisions[0].CanonicalID = s
		},
		"relation_mention": func(r *history.Record[string, string, attributes], s string) {
			r.Result.RelationDecisions[0].Mention = s
		},
		"relation_from": func(r *history.Record[string, string, attributes], s string) { r.Result.RelationDecisions[0].From = s },
		"relation_to":   func(r *history.Record[string, string, attributes], s string) { r.Result.RelationDecisions[0].To = s },
		"relation_key":  func(r *history.Record[string, string, attributes], s string) { r.Result.RelationDecisions[0].Key = s },
		"relation_canonical": func(r *history.Record[string, string, attributes], s string) {
			r.Result.RelationDecisions[0].CanonicalID = s
		},
		"ambiguous_from": func(r *history.Record[string, string, attributes], s string) { r.Result.RelationDecisions[1].From = s },
		"ambiguous_to":   func(r *history.Record[string, string, attributes], s string) { r.Result.RelationDecisions[1].To = s },
	}
}

func TestValidReplacementCharacterHistoryRoundTrip(t *testing.T) {
	// Arrange: U+FFFD is valid identity, optional namespace and parent remain absent.
	input := record(t, "r1", "TeamA", "�", "")
	input.Metadata.Run, input.Metadata.ExtractionFingerprint = "�", "�"
	input.Input.Entities[0].Name = "�"
	// Act.
	snapshot, err := history.Capture(
		context.Background(),
		access.Unrestricted(),
		input,
		admitted,
		maxBytes,
		maxSupports,
	)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := snapshot.Record()
	// Assert: faithful declared record; no core normalization or policy re-execution.
	if err != nil || !reflect.DeepEqual(input, restored) {
		t.Fatal(restored, err)
	}
}
