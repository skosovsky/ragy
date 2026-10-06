//go:build darwin || linux

package pdf_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func captureDurableLayoutEvidence(
	t *testing.T,
	schema filter.Schema,
	read access.Binding,
	results retrieval.ResultSet[sourceMeta],
	host *layoutPayloadHost,
) (evidence.Record, evidence.Input[sourceMeta], evidence.Policy) {
	t.Helper()
	allowed := map[string]bool{
		"pdf-durable":                  true,
		"parsed-layout":                true,
		"dense":                        true,
		read.Snapshot().Identity:       true,
		read.Publication().Reference(): true,
	}
	hits, originals, allowedLocations, outcome, reason := layoutObservedHits(t, results, allowed)
	reader, readerErr := source.NewReader(source.ReadConfig[sourceMeta, layout.Retained]{
		Target: "layout", Schema: schema, Catalog: host, Loader: host,
		Attributes:      retrieval.NewJSONCodec[sourceMeta](schema).Encode,
		ValidatePayload: func(ref source.Reference, value layout.Retained) error { return value.Validate(ref) },
		ClonePayload:    layout.CloneRetained,
	})
	if readerErr != nil {
		t.Fatal(readerErr)
	}
	var allRefs []source.Reference
	for ref := range originals {
		allRefs = append(allRefs, ref)
	}
	input := evidence.Input[sourceMeta]{
		Schema:         schema,
		Codec:          retrieval.NewJSONCodec[sourceMeta](schema),
		RetrievalID:    "pdf-durable",
		RecipeRevision: "parsed-layout",
		Outcome:        outcome,
		Reason:         reason,
		Coverage:       retrieval.CompleteReadCoverage(),
		Required:       []evidence.Field{evidence.SourceField, evidence.ScoreField, evidence.PublicationField},
		Stages: []evidence.Stage[sourceMeta]{
			{
				Name:      "dense",
				Status:    evidence.StageObserved,
				Scores:    evidence.Observed,
				Sources:   evidence.Observed,
				Judgments: evidence.Unavailable,
				Hits:      hits,
			},
		},
		SourceAdmission: func(ctx context.Context, binding access.Binding, ref source.Reference) error {
			_, ok := originals[ref]
			if !ok {
				return ragy.ErrProtocol
			}
			// Admit the complete original batch before any payload load, including export.
			_, err := reader.Lookup(ctx, source.LookupRequest{Read: binding, References: allRefs})
			return err
		},
	}
	policy := evidence.Policy{
		AllowIdentifier: func(_ evidence.IdentifierKind, id string) bool { return allowed[id] },
		AllowNumbers:    true,
		AllowLocation:   func(loc source.Locator) bool { return allowedLocations[loc] },
	}
	record, err := evidence.Capture(t.Context(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = evidence.Decode(wire); err != nil {
		t.Fatal("strict actual PDF evidence", err)
	}
	missing := input
	missing.Stages = slices.Clone(input.Stages)
	missing.Stages[0].Scores = evidence.Unavailable
	missingRecord, missingErr := evidence.Capture(t.Context(), read, missing, policy)
	if !errors.Is(missingErr, ragy.ErrUnavailable) {
		t.Fatal("required missing score capability accepted", missingErr)
	}
	if _, missingErr = missingRecord.MarshalJSON(); missingErr == nil {
		t.Fatal("missing capability returned record")
	}
	t.Logf("TASK12_EVIDENCE %s", wire)
	snapshot, err := record.Snapshot()
	if err != nil || snapshot.Outcome != evidence.Partial || snapshot.Reason != evidence.MissingEvidence ||
		len(snapshot.Stages) != 1 ||
		len(snapshot.Stages[0].Hits) != results.Len() {
		t.Fatal("PDF coverage/observations lost", err)
	}
	// Act: mutate the producer's observed locator/document after immutable capture.
	before := bytes.Clone(wire)
	saved := input.Stages[0].Hits[0].Locations[0]
	input.Stages[0].Hits[0].Locations[0].Reference.Revision = "mutated"
	input.Stages[0].Hits[0].Document.Meta.Coverage = "complete"
	after, err := json.Marshal(record)
	// Assert: frozen wire remains partial and revision-bound.
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("PDF evidence aliases producer", err)
	}
	input.Stages[0].Hits[0].Locations[0] = saved
	input.Stages[0].Hits[0].Document.Meta.Coverage = "partial"
	return record, input, policy
}

func rejectRetiredLayoutEvidence(
	t *testing.T,
	read access.Binding,
	input evidence.Input[sourceMeta],
	policy evidence.Policy,
	host *layoutPayloadHost,
) {
	t.Helper()
	before := host.payloadCalls
	// Act: the same previously observed original source must be admitted again at export.
	record, err := evidence.Capture(t.Context(), read, input, policy)
	// Assert: a deleted/denied r1 never becomes a recorded r2 or partial leaking record.
	if !errors.Is(err, ragy.ErrUnavailable) || host.payloadCalls != before {
		t.Fatal("unavailable PDF original leaked during recording", err)
	}
	if _, err = record.MarshalJSON(); err == nil {
		t.Fatal("rejected capture returned record")
	}
}

func layoutObservedHits(
	t *testing.T,
	results retrieval.ResultSet[sourceMeta],
	allowed map[string]bool,
) ([]evidence.Hit[sourceMeta], map[source.Reference]source.Locator, map[source.Locator]bool, evidence.Outcome, evidence.Reason) {
	t.Helper()
	var hits []evidence.Hit[sourceMeta]
	originals := make(map[source.Reference]source.Locator)
	allowedLocations := make(map[source.Locator]bool)
	outcome, reason := evidence.Complete, evidence.NoReason
	for _, doc := range results.Documents() {
		if doc.Meta.Coverage != "partial" {
			t.Fatal("partial OCR coverage lost before recording")
		}
		outcome, reason = evidence.Partial, evidence.MissingEvidence
		// Indexed dense-vector identity is not an original PDF citation.
		locs := doc.SourceMapping.Supports()
		doc.SourceSupports = locs
		var refs []source.Reference
		seen := make(map[source.Reference]bool)
		for _, loc := range locs {
			originals[loc.Reference] = loc
			allowedLocations[loc] = true
			if !seen[loc.Reference] {
				refs = append(refs, loc.Reference)
				seen[loc.Reference] = true
			}
			for _, id := range []string{loc.Reference.Namespace, loc.Reference.Source, loc.Reference.Revision, loc.Reference.Transformation, loc.Reference.Artifact, loc.Reference.Representation, loc.Cell.Table, loc.Cell.Element, loc.Page.PrintedLabel} {
				allowed[id] = true
			}
		}
		allowed[doc.ID] = true
		allowed[string(doc.ScoreSemantics)] = true
		hits = append(hits, evidence.Hit[sourceMeta]{Document: doc, Sources: refs, Locations: locs})
	}
	return hits, originals, allowedLocations, outcome, reason
}
