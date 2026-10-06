//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"path/filepath"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/graphingest/resolution/history"
)

func archiveResolution(
	ctx context.Context,
	read access.Binding,
	root string,
	f fixture,
	batches []sourceExtraction,
	input resolution.Extraction[string, string, graphAttributes],
	result resolution.Result[string, string, graphAttributes],
	parent string,
) (history.Reference, error) {
	var configurations []string
	for _, row := range f.Sources {
		configurations = append(configurations, sourceConfiguration(batches, row.ID))
	}
	encoded, err := json.Marshal(configurations)
	if err != nil {
		return history.Reference{}, err
	}
	fingerprint := digest(encoded)
	snapshot, err := history.Capture(ctx, read, history.Record[string, string, graphAttributes]{
		Metadata: history.Metadata{
			Run:                   "capture:" + fingerprint,
			ExtractionFingerprint: fingerprint,
			Parent:                parent,
		},
		Input:  input,
		Result: result,
	}, originalAdmission(f), baselineFileBytes, localEdgeCap)
	if err != nil {
		return history.Reference{}, err
	}
	path := filepath.Join(root, "resolution-history")
	store, err := history.NewFileStore[string, string, graphAttributes](
		path,
		baselineFileBytes,
		localEdgeCap,
		originalAdmission(f),
	)
	if err != nil {
		return history.Reference{}, err
	}
	if err = store.Append(ctx, read, snapshot); err != nil {
		return history.Reference{}, err
	}
	reopened, err := history.NewFileStore[string, string, graphAttributes](
		path,
		baselineFileBytes,
		localEdgeCap,
		originalAdmission(f),
	)
	if err != nil {
		return history.Reference{}, err
	}
	retained, err := reopened.Read(ctx, read, snapshot.Reference())
	if err != nil {
		return history.Reference{}, err
	}
	record, err := retained.Record()
	if err != nil {
		return history.Reference{}, err
	}
	if record.Metadata.ExtractionFingerprint != fingerprint || record.Result.PolicyIdentity != result.PolicyIdentity ||
		record.Result.OntologyIdentity != result.OntologyIdentity {
		return history.Reference{}, errInvalid
	}
	return retained.Reference(), nil
}
