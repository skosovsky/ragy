//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphexpand"
	"github.com/skosovsky/ragy/source"
)

type task19GraphMeta struct {
	Scope    string `json:"scope"`
	Document string `json:"document"`
}
type task19GraphDocument struct {
	ID        string                `json:"id"`
	Scope     string                `json:"scope"`
	Text      string                `json:"text"`
	Reference source.Reference      `json:"reference"`
	Relations []task19GraphRelation `json:"relations"`
}
type task19GraphRelation struct {
	TargetID string `json:"target_id"`
	Type     string `json:"type"`
}
type task19GraphExecution struct {
	IDs         []string
	References  []source.Reference
	Publication string
	Calls       uint64
}

type task19PreparedGraph struct {
	adapter     *managed.Adapter[task19GraphMeta]
	schema      filter.Schema
	scopeField  filter.Field[string]
	publication access.Publication
	documents   map[string]task19GraphDocument
	recordCap   int
}

func task19GraphKey(namespace, scope string, docs []task19GraphDocument) string {
	raw, _ := json.Marshal(docs)
	return namespace + "/" + scope + "/" + digest(raw)
}

func task19UsePrepared(
	ctx context.Context,
	p task19PreparedGraph,
	scope string,
	seeds []string,
	slots uint64,
	duration time.Duration,
) (task19GraphExecution, error) {
	out := task19GraphExecution{Publication: p.publication.Reference()}
	read, err := task19GraphBinding(p.schema, p.scopeField, p.publication, scope, duration)
	if err != nil {
		return out, err
	}
	return task19TraverseGraph(ctx, p.adapter, read, scope, p.documents, seeds, slots, duration, out)
}

// task19Expand executes actual managed source-supported graph expansion; corpus
// relations are authored facts and this local profile calls no extraction model.
func task19Expand(
	ctx context.Context,
	preparedGraphs map[string]task19PreparedGraph,
	root, namespace, scope string,
	docs []task19GraphDocument,
	seeds []string,
	slots uint64,
	duration time.Duration,
) (task19GraphExecution, error) {
	var out task19GraphExecution
	key := task19GraphKey(namespace, scope, docs)
	if prepared, ok := preparedGraphs[key]; ok {
		return task19UsePrepared(ctx, prepared, scope, seeds, slots, duration)
	}
	fields := filter.NewSchema()
	sf, err := fields.String("scope")
	if err != nil {
		return out, err
	}
	if _, err = fields.String("document"); err != nil {
		return out, err
	}
	schema, err := fields.Build()
	if err != nil {
		return out, err
	}
	store, err := filestore.New(root, baselineManifestBytes)
	if err != nil {
		return out, err
	}
	recordCap := 1
	byID := map[string]task19GraphDocument{}
	for _, d := range docs {
		byID[d.ID] = d
		recordCap += 1 + 2*len(d.Relations)
	}
	adapter, err := managed.New(
		managed.Config[task19GraphMeta]{
			Namespace:           namespace,
			Target:              graphTarget,
			Store:               store,
			Schema:              graph.Schema{NodeAttributes: schema, EdgeAttributes: schema},
			MaxRecords:          recordCap,
			MaxAdmissionRecords: recordCap,
			CloneMeta:           func(m task19GraphMeta) (task19GraphMeta, error) { return m, nil },
		},
	)
	if err != nil {
		return out, err
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[managed.Payload[task19GraphMeta]]{
			Store:   store,
			Targets: []lifecycle.Registration[managed.Payload[task19GraphMeta]]{{Name: graphTarget, Port: adapter}},
			Now:     time.Now,
			ClonePayload: func(p managed.Payload[task19GraphMeta]) (managed.Payload[task19GraphMeta], error) {
				p.Nodes = slices.Clone(p.Nodes)
				p.Edges = slices.Clone(p.Edges)
				for i := range p.Nodes {
					p.Nodes[i].Value.Labels = slices.Clone(p.Nodes[i].Value.Labels)
				}
				return p, nil
			},
			ValidatePayload: func(m lifecycle.Manifest, p managed.Payload[task19GraphMeta]) error {
				raw, e := json.Marshal(p)
				if e != nil {
					return e
				}
				if digest(raw) != m.Payload {
					return errInvalid
				}
				return nil
			},
		},
	)
	if err != nil {
		return out, err
	}
	for _, d := range docs {
		if err = task19PublishGraph(ctx, executor, namespace, d, byID); err != nil {
			return out, err
		}
	}

	pub, err := lifecycle.CapturePublication(ctx, store, namespace, []string{graphTarget})
	if err != nil {
		return out, err
	}
	out.Publication = pub.Reference()
	prepared := task19PreparedGraph{
		adapter:     adapter,
		schema:      schema,
		scopeField:  sf,
		publication: pub,
		documents:   byID,
		recordCap:   recordCap,
	}
	preparedGraphs[key] = prepared
	return task19UsePrepared(ctx, prepared, scope, seeds, slots, duration)
}

func task19TraverseGraph(
	ctx context.Context,
	adapter *managed.Adapter[task19GraphMeta],
	read access.Binding,
	scope string,
	byID map[string]task19GraphDocument,
	seeds []string,
	slots uint64,
	duration time.Duration,
	out task19GraphExecution,
) (task19GraphExecution, error) {
	now := time.Now()
	if len(seeds) == 0 {
		return out, read.Check(ctx)
	}
	ledger, err := budget.New(
		budget.Config{
			Limits:           budget.Limits{RetrievalCalls: slots, ModelCalls: 0},
			Deadline:         now.Add(duration),
			Now:              time.Now,
			RequireKnownCost: true,
		},
	)
	if err != nil {
		return out, err
	}
	exp, err := graphexpand.New(
		graphexpand.Config[task19GraphMeta]{
			Adapter:  adapter,
			MaxDepth: 1,
			MaxNodes: task19.CandidateLimit,
			MaxEdges: task19.CandidateLimit,
			Duration: duration,
			Now:      time.Now,
			Quote: func(context.Context) (budget.Reservation, error) {
				return budget.Reservation{Kind: budget.Retrieval, CostKnown: true}, nil
			},
			CloneMeta: func(m task19GraphMeta) (task19GraphMeta, error) { return m, nil },
		},
	)
	if err != nil {
		return out, err
	}
	result, err := exp.Run(
		ctx,
		graphexpand.Request{
			Read: read,
			Traversal: graph.TraversalRequest{
				Seeds:     slices.Clone(seeds),
				Direction: graph.DirectionUndirected,
				Depth:     1,
			},
		},
		ledger,
	)
	out.Calls = result.GraphCalls
	out.References = result.SourceReferences()
	if err != nil {
		return out, err
	}

	return task19ProjectGraph(ctx, read, scope, byID, result, out)
}

func task19ProjectGraph(
	ctx context.Context,
	read access.Binding,
	scope string,
	byID map[string]task19GraphDocument,
	result graphexpand.Result[task19GraphMeta],
	out task19GraphExecution,
) (task19GraphExecution, error) {
	// Graph reachability supplies no numeric similarity. Exact original supports
	// and mandatory scope are required before mapping nodes to delivered documents.
	for _, n := range result.Evidence.Snapshot.Nodes {
		d, ok := byID[n.ID]
		originalAdmitted := false
		for _, support := range result.Evidence.Supports {
			if support.Kind == "node" && support.ID == n.ID && slices.Contains(support.References, d.Reference) {
				originalAdmitted = true
				break
			}
		}
		if !ok || (d.Scope != "public" && d.Scope != scope) || !originalAdmitted {
			return out, errInvalid
		}
		out.IDs = append(out.IDs, d.ID)
	}
	return out, read.Check(ctx)
}

func task19PublishGraph(
	ctx context.Context,
	executor *lifecycle.Executor[managed.Payload[task19GraphMeta]],
	namespace string,
	d task19GraphDocument,
	byID map[string]task19GraphDocument,
) error {
	ref := d.Reference
	ref.Transformation = "task19-authored-graph-v1"
	ref.Artifact = d.ID
	ref.Representation = "graph-node"
	payload := managed.Payload[task19GraphMeta]{
		Nodes: []managed.Node[task19GraphMeta]{
			{
				Reference: ref,
				Value: graph.Node[task19GraphMeta]{
					ID:      d.ID,
					Labels:  []string{"Document"},
					Content: d.ID,
					Meta:    task19GraphMeta{Scope: d.Scope, Document: d.ID},
				},
			},
		},
	}
	artifacts := []lifecycle.Artifact{{Reference: ref, Supports: []source.Reference{d.Reference}}}
	for i, rel := range d.Relations {
		target, ok := byID[rel.TargetID]
		if !ok {
			continue
		}
		// Each source payload must contain both edge endpoints. The
		// source-authored relation supports an endpoint mention; the
		// target document's own published node contributes its original.
		endpointExists := false
		for _, n := range payload.Nodes {
			if n.Value.ID == target.ID {
				endpointExists = true
				break
			}
		}
		if !endpointExists {
			tr := ref
			tr.Artifact = target.ID
			payload.Nodes = append(
				payload.Nodes,
				managed.Node[task19GraphMeta]{
					Reference: tr,
					Value: graph.Node[task19GraphMeta]{
						ID:      target.ID,
						Labels:  []string{"Document"},
						Content: target.ID,
						Meta:    task19GraphMeta{Scope: target.Scope, Document: target.ID},
					},
				},
			)
			artifacts = append(artifacts, lifecycle.Artifact{Reference: tr, Supports: []source.Reference{d.Reference}})
		}
		er := ref
		er.Artifact = fmt.Sprintf("%s/edge/%d", d.ID, i)
		er.Representation = "graph-edge"
		payload.Edges = append(
			payload.Edges,
			managed.Edge[task19GraphMeta]{
				Reference: er,
				Value: graph.Edge[task19GraphMeta]{
					ID:       er.Artifact,
					SourceID: d.ID,
					TargetID: rel.TargetID,
					Type:     rel.Type,
					Meta:     task19GraphMeta{Scope: d.Scope, Document: d.ID},
				},
			},
		)
		artifacts = append(artifacts, lifecycle.Artifact{Reference: er, Supports: []source.Reference{d.Reference}})
	}
	raw, e := json.Marshal(payload)
	if e != nil {
		return e
	}
	m := lifecycle.Manifest{
		ID:      d.ID,
		Key:     d.ID,
		Payload: digest(raw),
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      ref.Namespace,
			Source:         ref.Source,
			Revision:       ref.Revision,
			Content:        digest([]byte(d.Text)),
			Transformation: ref.Transformation,
			Access:         ref.AccessFingerprint,
		},
		Targets: []lifecycle.Target{
			{Name: graphTarget, Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
	return task19ApplyGraph(ctx, executor, namespace, d.ID, m, payload)
}

func task19ApplyGraph(
	ctx context.Context,
	executor *lifecycle.Executor[managed.Payload[task19GraphMeta]],
	namespace, id string,
	m lifecycle.Manifest,
	payload managed.Payload[task19GraphMeta],
) error {
	if _, err := executor.Prepare(ctx, m); err != nil {
		return err
	}
	if _, err := executor.Stage(ctx, namespace, id, graphTarget, payload); err != nil {
		return err
	}
	if _, err := executor.Publish(ctx, namespace, id); err != nil {
		return err
	}
	return nil
}

func task19GraphBinding(
	schema filter.Schema,
	sf filter.Field[string],
	pub access.Publication,
	scope string,
	duration time.Duration,
) (access.Binding, error) {
	b, err := filter.NewBuilder(schema)
	if err != nil {
		return access.Binding{}, err
	}
	mandatory, err := filter.In(b, sf, "public", scope).Build()
	if err != nil {
		return access.Binding{}, err
	}
	now := time.Now()
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "task19-scope:" + scope,
				PolicyEpoch: 1,
				IssuedAt:    now,
				ExpiresAt:   now.Add(duration),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: pub,
			Now:         time.Now,
			Authority:   access.AuthorityFunc(func(ctx context.Context, _ access.Snapshot) error { return ctx.Err() }),
		},
	)

	return read, err
}
