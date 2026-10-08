//go:build e2e && (darwin || linux)

package pdf_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"maps"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func layoutPayloadDigest(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
func layoutDenseSpace() dense.Space {
	return dense.Space{Metric: "normalized-dot",
		Model:         "fixture",
		ModelRevision: "fixed",
		Configuration: "saved-vectors",
		VectorSpace:   "dot",
		Dimension:     2,
	}
}

func publishParsedLayout(
	t *testing.T,
	cfg persistent.Config[sourceMeta],
	revision, expected string,
) (*layoutPayloadHost, []persistent.Record[sourceMeta]) {
	t.Helper()
	parsed := parseLayoutRevision(t, revision)
	ref := fixtureReference()
	_, projectRead := parserScope(t)
	projected, err := layout.Project(t.Context(), parsed, layout.ProjectionOptions{Read: projectRead,
		ImageText: func(_ context.Context, _ access.Binding, image layout.Image) (source.MappedText, error) {
			if image.Location.Page.PhysicalIndex != 0 {
				return image.OCR.Mapping()
			}
			return source.DerivedText("fixture diagram", []source.Locator{image.Location})
		}})
	if err != nil {
		t.Fatal(err)
	}
	records := layoutDenseRecords(projected)
	adapter, err := persistent.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]persistent.Record[sourceMeta]]{
		Store:   cfg.Store,
		Targets: []lifecycle.Registration[[]persistent.Record[sourceMeta]]{{Name: "dense", Port: adapter}},
		ClonePayload: func(values []persistent.Record[sourceMeta]) ([]persistent.Record[sourceMeta], error) {
			out := slices.Clone(values)
			for i := range out {
				out[i].Value.Vector = slices.Clone(out[i].Value.Vector)
			}
			return out, nil
		},
		ValidatePayload: func(m lifecycle.Manifest, values []persistent.Record[sourceMeta]) error {
			if layoutPayloadDigest(values) != m.Payload {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	manifest := lifecycle.Manifest{
		ID:                  revision,
		Key:                 revision,
		Payload:             layoutPayloadDigest(records),
		ExpectedPublication: expected,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace: ref.Namespace,
			Source:    ref.Source,
			Revision:  revision,
			Content: layoutPayloadDigest(
				loadFixture(t),
			),
			Transformation: "dense-layout",
			Access:         ref.AccessFingerprint,
		},
	}
	var artifacts []lifecycle.Artifact
	for _, record := range records {
		var supports []source.Reference
		for _, loc := range record.SourceMapping.Supports() {
			if !slices.Contains(supports, loc.Reference) {
				supports = append(supports, loc.Reference)
			}
		}
		artifacts = append(artifacts, lifecycle.Artifact{Reference: record.Reference, Supports: supports})
	}
	manifest.Targets = []lifecycle.Target{
		{Name: "dense", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
	}
	if _, err = executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(t.Context(), ref.Namespace, revision, "dense", records); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), ref.Namespace, revision); err != nil {
		t.Fatal(err)
	}
	return newModalityRetention(t, parsed, projected), records
}
func layoutDenseRecords(projected []layout.Projected) []persistent.Record[sourceMeta] {
	var records []persistent.Record[sourceMeta]
	for _, item := range projected {
		indexed := item.Location.Reference
		indexed.Transformation = "dense-layout"
		indexed.Artifact = item.ID
		indexed.Representation = "dense-vector"
		records = append(
			records,
			persistent.Record[sourceMeta]{
				Reference:     indexed,
				SourceMapping: item.Text,
				Value: dense.Record[sourceMeta]{Space: layoutDenseSpace(),
					ID:      item.ID,
					Content: item.Text.Text(),
					Vector:  []float32{1, 0},
					Meta: sourceMeta{
						Organization: "a",
						Visibility:   "public",
						Artifact:     item.Location.Reference.Artifact,
						Revision:     indexed.Revision,
						Coverage:     string(item.Coverage),
					},
				},
			},
		)
	}
	return records
}

func bindParsedLayout(
	t *testing.T,
	cfg persistent.Config[sourceMeta],
	host *layoutPayloadHost,
	revision string,
) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(t.Context(), cfg.Store, "manuals", []string{"dense"})
	if err != nil {
		t.Fatal(err)
	}
	targets := publication.Targets()
	for ref := range host.records {
		if ref.Revision != revision {
			continue
		}
		target := access.TargetRevision{
			Target:            "layout",
			Namespace:         ref.Namespace,
			Source:            ref.Source,
			Revision:          ref.Revision,
			Transformation:    ref.Transformation,
			AccessFingerprint: ref.AccessFingerprint,
		}
		if !slices.Contains(targets, target) {
			targets = append(targets, target)
		}
	}
	publication, err = access.PinPublication("layout-"+publication.Reference(), targets)
	if err != nil {
		t.Fatal(err)
	}
	_, scope := parserScope(t)
	mandatory, err := scope.Prepare(
		t.Context(),
		cfg.Schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot:    scope.Snapshot(),
		Schema:      cfg.Schema,
		Mandatory:   mandatory,
		Publication: publication,
		Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
		Now:         func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	return read
}
func TestE2EPDFLayoutDurablePublicationReopenAndRetainedRevision(t *testing.T) {
	// Arrange: real PDF projection, typed partial coverage, persistent files and CAS ledger.
	schema, _ := parserScope(t)
	ledgerRoot := filepath.Join(t.TempDir(), "ledger")
	store, err := filestore.New(ledgerRoot, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	cfg := persistent.Config[sourceMeta]{
		Root:            t.TempDir(),
		Namespace:       "manuals",
		Target:          "dense",
		Store:           store,
		Schema:          schema,
		Space:           layoutDenseSpace(),
		CloneMeta:       func(m sourceMeta) (sourceMeta, error) { return m, nil },
		MaxRecords:      100,
		MaxScanRecords:  100,
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
	}
	retained, records := publishParsedLayout(t, cfg, "r1", "")
	oldRead := bindParsedLayout(t, cfg, retained, "r1")
	newer, _ := publishParsedLayout(t, cfg, "r2", "r1")
	maps.Copy(retained.records, newer.records)
	maps.Copy(retained.payloads, newer.payloads)
	cfg.Store, err = filestore.New(ledgerRoot, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	// Act: fresh adapter loads saved r1 through its retained publication despite active r2.
	reopened, err := persistent.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reopened.Retrieve(t.Context(), retrieval.Query[persistent.Intent]{
		Read: oldRead,
		Intent: persistent.Intent{
			Embedding: dense.Embedding{Space: layoutDenseSpace(), Vector: []float32{1, 0}},
		},
		Options: retrieval.RetrieveOptions{TopK: 100},
	})
	if err != nil || results.Len() != len(records) {
		t.Fatal(results, err)
	}
	verifyDurableLayoutResults(t, cfg, oldRead, results, retained)
}

func verifyDurableLayoutResults(
	t *testing.T,
	cfg persistent.Config[sourceMeta],
	read access.Binding,
	results retrieval.ResultSet[sourceMeta],
	host *layoutPayloadHost,
) {
	t.Helper()
	resolver, err := layout.NewResolver(
		layout.ResolverConfig[sourceMeta]{Target: "layout", Schema: cfg.Schema, Catalog: host, Loader: host,
			Attributes: retrieval.NewJSONCodec[sourceMeta](cfg.Schema).Encode},
	)
	if err != nil {
		t.Fatal(err)
	}
	locations, image := durableOriginalLocations(t, results)
	resolved, err := resolver.Resolve(t.Context(), layout.ResolveRequest{Read: read, Locations: locations})
	if err != nil || len(resolved) == 0 || image == nil {
		t.Fatal(resolved, err)
	}
	_, recordInput, recordPolicy := captureDurableLayoutEvidence(t, cfg.Schema, read, results, host)
	// Deleted r1 while r2 is retained fails before payload loading; no latest substitution.
	originalRow := host.records[image.Reference]
	newerRef := image.Reference
	newerRef.Revision = "r2"
	if _, exists := host.records[newerRef]; !exists {
		t.Fatal("missing retained r2")
	}
	delete(host.records, image.Reference)
	before := host.payloadCalls
	denied, err := resolver.Resolve(t.Context(), layout.ResolveRequest{Read: read, Locations: []source.Locator{*image}})
	if !errors.Is(err, ragy.ErrUnavailable) || len(denied) != 0 || host.payloadCalls != before {
		t.Fatal(denied, err)
	}
	rejectRetiredLayoutEvidence(t, read, recordInput, recordPolicy, host)
	originalRow.meta.Visibility = "private"
	host.records[image.Reference] = originalRow
	denied, err = resolver.Resolve(t.Context(), layout.ResolveRequest{Read: read, Locations: []source.Locator{*image}})
	if !errors.Is(err, ragy.ErrUnavailable) || len(denied) != 0 || host.payloadCalls != before {
		t.Fatal("denied r1 loaded", denied, err)
	}
	rejectRetiredLayoutEvidence(t, read, recordInput, recordPolicy, host)
}

func durableOriginalLocations(
	t *testing.T,
	results retrieval.ResultSet[sourceMeta],
) ([]source.Locator, *source.Locator) {
	t.Helper()
	var locations []source.Locator
	var image *source.Locator
	for _, doc := range results.Documents() {
		// Assert: exact original geometry/revision and document partial coverage persist across restart.
		if doc.Meta.Revision != "r1" || doc.Meta.Coverage != "partial" || doc.SourceMapping.Text() != doc.Content {
			t.Fatal(doc)
		}
		for _, loc := range doc.SourceMapping.Supports() {
			assertParsedLocatorEnvelope(t, loc)
			if loc.Reference.Revision != "r1" || loc.Reference.Representation == "dense-vector" {
				t.Fatal(loc)
			}
			locations = append(locations, loc)
			if loc.Kind == source.ImageLocation {
				value := loc
				image = &value
				if doc.SourceMapping.Fragments()[0].Origin != source.DerivedContent {
					t.Fatal("image prose became original")
				}
			}
		}
	}
	return locations, image
}

func parseLayoutRevision(t *testing.T, revision string) layout.Document {
	t.Helper()
	parser, err := pdfadapter.New(parserConfig(integrationPython(t)))
	if err != nil {
		t.Fatal(err)
	}
	ref := fixtureReference()
	ref.Revision = revision
	parsed, err := parser.Parse(t.Context(), layout.Input{Reference: ref, Data: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err = layout.ApplyOCR(t.Context(), parsed, []layout.OCRObservation{
		{
			Source:         parsed.Pages[1].Images[0].Location,
			State:          layout.OCRUnreadable,
			Transformation: "fixture-ocr-simulation",
		},
	})
	if err != nil || parsed.Coverage != layout.Partial ||
		parsed.Pages[1].Diagnostics[0].Code != layout.DiagnosticOCRUnreadable {
		t.Fatal("partial OCR observation", err)
	}
	return parsed
}

func assertParsedLocatorEnvelope(t *testing.T, loc source.Locator) {
	t.Helper()
	encoded, err := source.EncodeLocator(loc)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := source.DecodeLocator(encoded)
	if err != nil || decoded != loc {
		t.Fatal("parser locator envelope", decoded, err)
	}
	original, err := loc.Identity()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := decoded.Identity()
	if err != nil || restored != original {
		t.Fatal("parser citation identity", err)
	}
}
