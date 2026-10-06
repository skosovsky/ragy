//go:build darwin || linux

package main

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/skosovsky/ragy/graphingest/resolution/history"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func TestGraphModelConfigurationChangesPublicationAndRetainsHistory(t *testing.T) {
	// Arrange: identical input facts with two declared model configurations.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	dense, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	original := deterministicSourceExtractions(t, f)
	changed := slices.Clone(original)
	for i := range changed {
		changed[i].Configuration = bindTokenizerIdentity(
			providerExtractionIdentity("second-model", ""),
			"qualified-counter-b",
		)
	}
	// Act: actual durable publication plus immutable history for both configurations.
	first, err := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, original)
	if err != nil {
		t.Fatal(err)
	}
	second, err := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, changed)
	if err != nil {
		t.Fatal(err)
	}
	firstTargets, err := first.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	secondTargets, err := second.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	// Assert: canonical facts/supports stay exact, transformation/publication changes.
	if slices.Equal(firstTargets, secondTargets) || first.history.ID == second.history.ID {
		t.Fatal(firstTargets, secondTargets)
	}
	assertProviderGraphGold(t, dense, first, f)
	assertProviderGraphGold(t, dense, second, f)
	assertSharedHistoryRetained(t, dense, f, original, changed, first, second)
}

func assertSharedHistoryRetained(
	t *testing.T,
	dense denseCorpus,
	f fixture,
	original, changed []sourceExtraction,
	first, second graphCorpus,
) {
	t.Helper()
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	input, err := combineExtractions(f, original)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	a, err := archiveResolution(t.Context(), read, root, f, original, input, first.resolved, "")
	if err != nil {
		t.Fatal(err)
	}
	b, err := archiveResolution(t.Context(), read, root, f, changed, input, second.resolved, a.ID)
	if err != nil {
		t.Fatal(err)
	}
	store, err := history.NewFileStore[string, string, graphAttributes](
		filepath.Join(root, "resolution-history"),
		baselineFileBytes,
		localEdgeCap,
		originalAdmission(f),
	)
	if err != nil {
		t.Fatal(err)
	}
	old, err := store.Read(t.Context(), read, a)
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := store.Read(t.Context(), read, b)
	if err != nil {
		t.Fatal(err)
	}
	oldRecord, err := old.Record()
	if err != nil {
		t.Fatal(err)
	}
	newRecord, err := fresh.Record()
	if err != nil {
		t.Fatal(err)
	}
	if a.ID == b.ID || oldRecord.Metadata.ExtractionFingerprint == newRecord.Metadata.ExtractionFingerprint ||
		oldRecord.Metadata.Parent != "" || newRecord.Metadata.Parent != a.ID ||
		!slices.Equal(a.Supports, b.Supports) {
		t.Fatal(oldRecord.Metadata, newRecord.Metadata)
	}
}
func TestProviderConfigurationIdentityBindsModelTokenizerAndRequestControls(t *testing.T) {
	// Arrange.
	base := providerExtractionIdentity("model-a", "")
	// Act/Assert: each declared model, endpoint and qualified counter partitions provenance.
	if base == providerExtractionIdentity("model-b", "") ||
		base == providerExtractionIdentity("model-a", "http://host.example") ||
		bindTokenizerIdentity(base, "counter-a") == bindTokenizerIdentity(base, "counter-b") {
		t.Fatal("configuration collision")
	}
	if !validFingerprint(base) || validFingerprint("unbound") || validFingerprint("") {
		t.Fatal("invalid configuration admission")
	}
}
func TestCaptureRejectsDeclaredModelMismatchBeforeDispatch(t *testing.T) {
	// Arrange: capture metadata cannot claim a model different from the provider binding.
	factories := captureFactories{extraction: func(context.Context) (extractionPorts, error) {
		return extractionPorts{modelIdentity: "actual-model"}, nil
	}, summary: func(context.Context) (summaryModelPorts, error) { return contractSummaryPorts(), nil }}
	// Act.
	raw, err := executeGraphCapture(
		t.Context(),
		t.TempDir(),
		captureIdentity{execution: contractExecution, model: "different-model"},
		factories,
	)
	// Assert.
	if err == nil || len(raw.Preparation.Extractions) != 0 || len(raw.Samples) != 0 {
		t.Fatal(raw, err)
	}
}

func TestSummaryDeclaredModelMismatchStopsBeforeDispatch(t *testing.T) {
	// Arrange: a prepared actual source corpus, with a model identity mismatch.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	factory := func(context.Context) (summaryModelPorts, error) {
		ports := contractSummaryPorts()
		ports.modelIdentity = "actual-summary-model"
		ports.model = func(context.Context, graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			calls++
			return graphsummary.ModelOutput{}, graphsummary.Usage{}, nil
		}
		return ports, nil
	}
	// Act.
	sample, err := captureGraphRecipe(
		t.Context(),
		corpus,
		prepared,
		read,
		f.Queries[1],
		captureIdentity{model: "different-model"},
		factory,
	)
	// Assert: no source payload or model callback runs under misleading metadata.
	if err == nil || calls != 0 || sample.ModelCalls != 0 || prepared.host.payloadCalls.Load() != 0 {
		t.Fatal(sample, err, calls)
	}
	a := providerSummaryIdentity("a", "")
	if !validFingerprint(a) || a == providerSummaryIdentity("b", "") {
		t.Fatal("unbound summary configuration")
	}
}
