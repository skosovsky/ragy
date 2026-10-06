//go:build darwin || linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"path/filepath"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
)

type meta struct {
	Tenant string `json:"tenant"`
}
type profiles struct {
	dense      *densefs.Adapter[meta]
	tensor     *tensorfs.Adapter[meta]
	denseRead  access.Binding
	tensorRead access.Binding
	refs       map[string]source.Reference
	denseRoot  string
	tensorRoot string
}

func reference(id, representation string) source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            "corpus",
		Revision:          "r1",
		Transformation:    "saved-embedding",
		AccessFingerprint: "acl",
		Artifact:          id,
		Representation:    representation,
	}
}

func publish[T any](
	ctx context.Context,
	store lifecycle.Store,
	target string,
	port lifecycle.StagePort[T],
	payload T,
	refs []source.Reference,
	clone func(T) (T, error),
) error {
	fingerprint := func(value T) (string, error) {
		data, err := json.Marshal(value)
		if err != nil {
			return "", err
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:]), nil
	}
	hash, err := fingerprint(payload)
	if err != nil {
		return err
	}
	artifacts := make([]lifecycle.Artifact, 0, len(refs))
	for _, ref := range refs {
		artifacts = append(artifacts, lifecycle.Artifact{Reference: ref, Supports: []source.Reference{ref}})
	}
	plan := lifecycle.Manifest{
		ID:      "operation",
		Key:     savedModel,
		Payload: hash,
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "corpus",
			Revision:       "r1",
			Content:        hash,
			Transformation: "saved-embedding",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: target, Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[T]{
			Store:        store,
			Targets:      []lifecycle.Registration[T]{{Name: target, Port: port}},
			ClonePayload: clone,
			ValidatePayload: func(manifest lifecycle.Manifest, value T) error {
				actual, e := fingerprint(value)
				if e != nil {
					return e
				}
				if actual != manifest.Payload {
					return ragy.ErrInvalidArgument
				}
				return nil
			},
		},
	)
	if err != nil {
		return err
	}
	if _, err = executor.Prepare(ctx, plan); err != nil {
		return err
	}
	if _, err = executor.Stage(ctx, "n", plan.ID, target, payload); err != nil {
		return err
	}
	_, err = executor.Publish(ctx, "n", plan.ID)
	return err
}

func pinned(
	ctx context.Context,
	store lifecycle.Store,
	target string,
	schema filter.Schema,
	tenant filter.Field[string],
) (access.Binding, error) {
	publication, err := lifecycle.CapturePublication(ctx, store, "n", []string{target})
	if err != nil {
		return access.Binding{}, err
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		return access.Binding{}, err
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		return access.Binding{}, err
	}
	now := time.Now()
	return access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "synthetic-scope",
				PolicyEpoch: 1,
				IssuedAt:    now,
				ExpiresAt:   now.Add(time.Minute),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: publication,
			Now:         time.Now,
			Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
		},
	)
}

func build(ctx context.Context, root string, input fixture) (profiles, error) {
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		return profiles{}, err
	}
	schema, err := fields.Build()
	if err != nil {
		return profiles{}, err
	}
	out := profiles{
		refs:       make(map[string]source.Reference),
		denseRoot:  filepath.Join(root, "dense"),
		tensorRoot: filepath.Join(root, "tensor"),
	}
	ds, err := filestore.New(filepath.Join(root, "dense-manifests"), manifestBytes)
	if err != nil {
		return profiles{}, err
	}
	ts, err := filestore.New(filepath.Join(root, "tensor-manifests"), manifestBytes)
	if err != nil {
		return profiles{}, err
	}
	denseSpace := dense.Space{
		Model:         savedModel,
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	tensorSpace := tensor.Space{
		Model:         savedModel,
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	dc := densefs.Config[meta]{
		Root:            out.denseRoot,
		Namespace:       "n",
		Target:          "dense",
		Store:           ds,
		Schema:          schema,
		Space:           denseSpace,
		CloneMeta:       func(m meta) (meta, error) { return m, nil },
		MaxRecords:      targetLimit,
		MaxScanRecords:  targetLimit,
		MaxCatalogBytes: targetFileBytes,
		MaxPayloadBytes: targetFileBytes,
	}
	tc := tensorfs.Config[meta]{
		Root:            out.tensorRoot,
		Namespace:       "n",
		Target:          "tensor",
		Store:           ts,
		Schema:          schema,
		Space:           tensorSpace,
		CloneMeta:       func(m meta) (meta, error) { return m, nil },
		MaxRecords:      targetLimit,
		MaxCatalogBytes: targetFileBytes,
		MaxPayloadBytes: targetFileBytes,
	}
	out.dense, err = densefs.New(dc)
	if err != nil {
		return profiles{}, err
	}
	out.tensor, err = tensorfs.New(tc)
	if err != nil {
		return profiles{}, err
	}
	dr, tr, drefs, trefs := records(input, denseSpace, tensorSpace)
	if err = publish(ctx, ds, "dense", out.dense, dr, drefs, cloneDense); err != nil {
		return profiles{}, err
	}
	if err = publish(ctx, ts, "tensor", out.tensor, tr, trefs, cloneTensor); err != nil {
		return profiles{}, err
	}
	// Fresh adapter instances must load the published catalog and payload from disk.
	out.dense, err = densefs.New(dc)
	if err != nil {
		return profiles{}, err
	}
	out.tensor, err = tensorfs.New(tc)
	if err != nil {
		return profiles{}, err
	}
	out.denseRead, err = pinned(ctx, ds, "dense", schema, tenant)
	if err != nil {
		return profiles{}, err
	}
	out.tensorRead, err = pinned(ctx, ts, "tensor", schema, tenant)
	for _, ref := range trefs {
		out.refs[ref.Artifact] = ref
	}
	return out, err
}

func records(
	input fixture,
	ds dense.Space,
	ts tensor.Space,
) ([]densefs.Record[meta], []tensorfs.Record[meta], []source.Reference, []source.Reference) {
	var dr []densefs.Record[meta]
	var tr []tensorfs.Record[meta]
	var drefs, trefs []source.Reference
	for _, doc := range input.Documents {
		dref, tref := reference(doc.ID, "dense-vector"), reference(doc.ID, "token-matrix")
		drefs = append(drefs, dref)
		trefs = append(trefs, tref)
		dr = append(
			dr,
			densefs.Record[meta]{
				Reference: dref,
				Space:     ds,
				Value:     dense.Record[meta]{ID: doc.ID, Content: doc.ID, Meta: meta{Tenant: "a"}, Vector: doc.Dense},
			},
		)
		tr = append(
			tr,
			tensorfs.Record[meta]{
				Reference: tref,
				Value: tensor.Record[meta]{
					ID:      doc.ID,
					Content: doc.ID,
					Meta:    meta{Tenant: "a"},
					Space:   ts,
					Tensor:  doc.Tokens,
				},
			},
		)
	}
	return dr, tr, drefs, trefs
}
func cloneDense(input []densefs.Record[meta]) ([]densefs.Record[meta], error) {
	out := slices.Clone(input)
	for i := range out {
		out[i].Value.Vector = slices.Clone(out[i].Value.Vector)
	}
	return out, nil
}
func cloneTensor(input []tensorfs.Record[meta]) ([]tensorfs.Record[meta], error) {
	out := slices.Clone(input)
	for i := range out {
		out[i].Value.Tensor = slices.Clone(out[i].Value.Tensor)
		for j := range out[i].Value.Tensor {
			out[i].Value.Tensor[j] = slices.Clone(out[i].Value.Tensor[j])
		}
	}
	return out, nil
}
