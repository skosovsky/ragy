//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"path/filepath"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/graphingest/resolution/history"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
)

const (
	graphTarget         = "graph"
	serviceKind         = "Service"
	billingName         = "Billing"
	productionNamespace = "prod"
)

type graphMetadata struct {
	Tenant   string `json:"tenant"`
	SourceID string `json:"source_key"`
	Owner    string `json:"owner,omitempty"`
}
type graphAttributes struct {
	Owner string `json:"owner"`
}
type sourceExtraction struct {
	Configuration string
	SourceID      string
	Value         resolution.Extraction[string, string, graphAttributes]
}
type graphCorpus struct {
	history  history.Reference
	adapter  *managed.Adapter[graphMetadata]
	store    lifecycle.Store
	resolved resolution.Result[string, string, graphAttributes]
}

func validateEntityKind(kind string, _ graphAttributes) error {
	if !slices.Contains([]string{serviceKind, "Database", "Team"}, kind) {
		return ragy.ErrInvalidGraph
	}
	return nil
}
func validateRelationKind(kind, from, to string, _ graphAttributes) error {
	if from != serviceKind || kind == "depends_on" && to != "Database" || kind == "owned_by" && to != "Team" ||
		!slices.Contains([]string{"depends_on", "owned_by"}, kind) {
		return ragy.ErrInvalidGraph
	}
	return nil
}
func entityDecision(e resolution.Entity[string, graphAttributes]) (resolution.Decision, error) {
	if e.Namespace == "" {
		return resolution.Decision{State: resolution.Ambiguous}, nil
	}
	name := e.Name
	if e.Namespace == productionNamespace && e.Kind == serviceKind && (name == "Pay" || name == billingName) {
		name = billingName
	}
	return resolution.Decision{
		State:     resolution.Resolved,
		Namespace: e.Namespace,
		Key:       e.Kind + "/" + name,
		Name:      name,
	}, nil
}
func originalAdmission(f fixture) func(context.Context, access.Binding, source.Locator) error {
	return func(ctx context.Context, read access.Binding, loc source.Locator) error {
		if err := read.Check(ctx); err != nil {
			return err
		}
		for _, row := range f.Sources {
			mapping, err := mappedSource(row)
			if err != nil {
				return err
			}
			if slices.Contains(mapping.Supports(), loc) {
				return nil
			}
		}
		return ragy.ErrUnavailable
	}
}

func combineExtractions(
	f fixture,
	batches []sourceExtraction,
) (resolution.Extraction[string, string, graphAttributes], error) {
	if len(batches) != len(f.Sources) {
		return resolution.Extraction[string, string, graphAttributes]{}, errInvalid
	}
	var combined resolution.Extraction[string, string, graphAttributes]
	seen := make(map[string]bool)
	for _, batch := range batches {
		exists := false
		var expectedNamespace string
		for _, row := range f.Sources {
			if row.ID == batch.SourceID {
				exists = true
				expectedNamespace = row.Namespace
			}
		}
		if !exists || seen[batch.SourceID] || !validFingerprint(batch.Configuration) {
			return combined, errInvalid
		}
		seen[batch.SourceID] = true
		for _, entity := range batch.Value.Entities {
			if entity.Namespace != expectedNamespace {
				return combined, errInvalid
			}
		}
		renamed, err := prefixExtraction(batch)
		if err != nil {
			return combined, err
		}
		combined.Entities = append(combined.Entities, renamed.Entities...)
		combined.Relations = append(combined.Relations, renamed.Relations...)
	}
	return combined, nil
}
func prefixExtraction(batch sourceExtraction) (resolution.Extraction[string, string, graphAttributes], error) {
	input := batch.Value
	input.Entities = slices.Clone(input.Entities)
	input.Relations = slices.Clone(input.Relations)
	names := make(map[string]string)
	for i, e := range input.Entities {
		if e.ID == "" || names[e.ID] != "" {
			return input, errInvalid
		}
		for _, loc := range e.Supports {
			if loc.Reference.Source != batch.SourceID {
				return input, errInvalid
			}
		}
		names[e.ID] = batch.SourceID + "/" + e.ID
		input.Entities[i].ID = names[e.ID]
		input.Entities[i].Supports = slices.Clone(e.Supports)
	}
	for i, e := range input.Relations {
		if names[e.From] == "" || names[e.To] == "" {
			return input, errInvalid
		}
		for _, loc := range e.Supports {
			if loc.Reference.Source != batch.SourceID {
				return input, errInvalid
			}
		}
		input.Relations[i].ID = batch.SourceID + "/" + e.ID
		input.Relations[i].From = names[e.From]
		input.Relations[i].To = names[e.To]
		input.Relations[i].Supports = slices.Clone(e.Supports)
	}
	return input, nil
}

func resolveGraph(
	ctx context.Context,
	read access.Binding,
	f fixture,
	input resolution.Extraction[string, string, graphAttributes],
) (resolution.Result[string, string, graphAttributes], error) {
	controls := referenceConfiguration()
	resolver, err := resolution.New(
		resolution.Config[string, string, graphAttributes]{
			OntologyIdentity: controls.Ontology,
			PolicyIdentity:   controls.IdentityPolicy,
			MaxEntities:      localNodeCap,
			MaxRelations:     localEdgeCap,
			MaxSupports:      localEdgeCap,
			ValidateEntity:   validateEntityKind,
			ValidateRelation: validateRelationKind,
			Identity:         entityDecision,
			RelationKey:      func(e resolution.Relation[string, graphAttributes]) (string, error) { return e.Kind, nil },
			CloneAttributes:  func(a graphAttributes) (graphAttributes, error) { return a, nil },
			Equivalent:       func(a, b graphAttributes) bool { return a == b },
			AdmitSupport:     originalAdmission(f),
		},
	)
	if err != nil {
		return resolution.Result[string, string, graphAttributes]{}, err
	}
	return resolver.Resolve(ctx, read, input)
}

func buildGraphCorpus(
	ctx context.Context,
	root string,
	denseCorpus denseCorpus,
	read access.Binding,
	batches []sourceExtraction,
) (graphCorpus, error) {
	combined, err := combineExtractions(denseCorpus.fixture, batches)
	if err != nil {
		return graphCorpus{}, err
	}
	resolved, err := resolveGraph(ctx, read, denseCorpus.fixture, combined)
	if err != nil {
		return graphCorpus{}, err
	}
	historyReference, err := archiveResolution(ctx, read, root, denseCorpus.fixture, batches, combined, resolved, "")
	if err != nil {
		return graphCorpus{}, err
	}
	schema, err := graphSourceSchema()
	if err != nil {
		return graphCorpus{}, err
	}
	store, err := filestore.New(filepath.Join(root, "graph-ledger"), baselineManifestBytes)
	if err != nil {
		return graphCorpus{}, err
	}
	adapter, err := managed.New(
		managed.Config[graphMetadata]{
			Namespace:  "n",
			Target:     graphTarget,
			Store:      store,
			Schema:     schema,
			MaxRecords: localEdgeCap,
			CloneMeta:  func(m graphMetadata) (graphMetadata, error) { return m, nil },
		},
	)
	if err != nil {
		return graphCorpus{}, err
	}
	builder, err := graphMaterializer(denseCorpus.fixture, schema)
	if err != nil {
		return graphCorpus{}, err
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[graphMetadata]]{
		Store: store,
		Targets: []lifecycle.Registration[managed.Payload[graphMetadata]]{
			{Name: graphTarget, Port: adapter},
		},
		ClonePayload: cloneGraphPayload,
		ValidatePayload: func(m lifecycle.Manifest, p managed.Payload[graphMetadata]) error {
			encoded, marshalErr := json.Marshal(p)
			if marshalErr != nil {
				return marshalErr
			}
			if len(m.Targets) != 1 || len(m.Targets[0].Artifacts) != len(p.Nodes)+len(p.Edges) ||
				m.Payload != digest(encoded) {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
		Now: time.Now,
	})
	if err != nil {
		return graphCorpus{}, err
	}
	for _, row := range denseCorpus.fixture.Sources {
		if publishErr := publishGraphSource(
			ctx,
			executor,
			builder,
			read,
			resolved,
			row,
			sourceConfiguration(batches, row.ID),
		); publishErr != nil {
			return graphCorpus{}, publishErr
		}
	}
	return graphCorpus{adapter: adapter, store: store, resolved: resolved, history: historyReference}, nil
}

func graphMaterializer(
	f fixture,
	schema graph.Schema,
) (*materialization.Materializer[string, string, graphAttributes, graphMetadata], error) {
	controls := referenceConfiguration()
	return materialization.New(
		materialization.Config[string, string, graphAttributes, graphMetadata]{
			OntologyIdentity: controls.Ontology,
			PolicyIdentity:   controls.IdentityPolicy,
			Schema:           schema,
			MaxFacts:         localEdgeCap,
			MaxSupports:      localEdgeCap,
			CloneAttributes:  func(a graphAttributes) (graphAttributes, error) { return a, nil },
			CloneMeta:        func(m graphMetadata) (graphMetadata, error) { return m, nil },
			Node: func(id resolution.Decision, kind string, attributes graphAttributes) (materialization.NodeValue[graphMetadata], error) {
				return materialization.NodeValue[graphMetadata]{
					Labels:  []string{kind},
					Content: id.Name,
					Meta:    graphMetadata{Tenant: "a", SourceID: id.Namespace + "/" + id.Key, Owner: attributes.Owner},
				}, nil
			},
			Edge: func(kind string, attributes graphAttributes) (materialization.EdgeValue[graphMetadata], error) {
				return materialization.EdgeValue[graphMetadata]{
					Type: kind,
					Meta: graphMetadata{Tenant: "a", SourceID: kind, Owner: attributes.Owner},
				}, nil
			},
			AdmitSupport: originalAdmission(f),
		},
	)
}
func cloneGraphPayload(input managed.Payload[graphMetadata]) (managed.Payload[graphMetadata], error) {
	input.Nodes = slices.Clone(input.Nodes)
	input.Edges = slices.Clone(input.Edges)
	for i := range input.Nodes {
		input.Nodes[i].Value.Labels = slices.Clone(input.Nodes[i].Value.Labels)
	}
	return input, nil
}
func (c graphCorpus) targets(ctx context.Context) ([]access.TargetRevision, error) {
	publication, err := lifecycle.CapturePublication(ctx, c.store, "n", []string{graphTarget})
	if err != nil {
		return nil, err
	}
	return publication.Targets(), nil
}

func publishGraphSource(
	ctx context.Context,
	executor *lifecycle.Executor[managed.Payload[graphMetadata]],
	builder *materialization.Materializer[string, string, graphAttributes, graphMetadata],
	read access.Binding,
	resolved resolution.Result[string, string, graphAttributes],
	row sourceRow,
	extractionConfiguration string,
) error {
	ref := originalReference(row.ID)
	plan, e := builder.Build(
		ctx,
		read,
		materialization.Request{
			Identity: lifecycle.Identity{
				Namespace:      ref.Namespace,
				Source:         ref.Source,
				Revision:       ref.Revision,
				Content:        digest([]byte(row.Text)),
				Transformation: "model-extraction:" + extractionConfiguration,
				Access:         ref.AccessFingerprint,
			},
			Target:             graphTarget,
			ManifestID:         row.ID,
			Key:                row.ID,
			PayloadFingerprint: digest([]byte(row.Text)),
		},
		resolved,
	)
	if e != nil {
		return e
	}
	encoded, encodeErr := json.Marshal(plan.Payload)
	if encodeErr != nil {
		return encodeErr
	}
	plan.Manifest.Payload = digest(encoded)
	if _, err := executor.Prepare(ctx, plan.Manifest); err != nil {
		return err
	}
	if _, err := executor.Stage(ctx, "n", row.ID, graphTarget, plan.Payload); err != nil {
		return err
	}
	if _, err := executor.Publish(ctx, "n", row.ID); err != nil {
		return err
	}
	return nil
}

func graphSourceSchema() (graph.Schema, error) {
	fields := filter.NewSchema()
	for _, name := range []string{"tenant", "source_key", "owner"} {
		if _, err := fields.String(name); err != nil {
			return graph.Schema{}, err
		}
	}
	attrs, err := fields.Build()
	if err != nil {
		return graph.Schema{}, err
	}
	return graph.NewSchema(attrs, attrs)
}

func sourceConfiguration(batches []sourceExtraction, id string) string {
	for _, batch := range batches {
		if batch.SourceID == id {
			return batch.Configuration
		}
	}
	return ""
}
