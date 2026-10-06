// Package history owns immutable JSON snapshots of typed resolution inputs and
// decisions. It is an optional persistence profile, not an ontology or workflow.
package history

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/source"
)

// Metadata binds recomputation to host extraction configuration and predecessor.
// Run and ExtractionFingerprint are nonempty valid UTF-8, without normalization.
// Parent is an earlier snapshot ID, not a mutable latest pointer or a CAS claim.
type Metadata struct {
	Run                   string `json:"run"`
	ExtractionFingerprint string `json:"extraction_fingerprint"`
	Parent                string `json:"parent"`
}
type Record[TKind, TRel comparable, TAttr any] struct {
	Metadata Metadata                                  `json:"metadata"`
	Input    resolution.Extraction[TKind, TRel, TAttr] `json:"input"`
	Result   resolution.Result[TKind, TRel, TAttr]     `json:"result"`
}

// Reference is portable metadata. Its supports form part of the storage filename,
// so a forged permitted support list cannot address a private payload by ID alone.
type Reference struct {
	ID       string           `json:"id"`
	Supports []source.Locator `json:"supports"`
}
type Admission func(context.Context, access.Binding, source.Locator) error

// Snapshot keeps serialized state private. Every Record/Reference call owns its
// output; callers cannot mutate the captured record through BYOT slice/map aliases.
type Snapshot[TKind, TRel comparable, TAttr any] struct {
	data      []byte
	reference Reference
}

func Capture[TKind, TRel comparable, TAttr any](
	ctx context.Context, read access.Binding, record Record[TKind, TRel, TAttr],
	admit Admission, maxBytes, maxSupports int,
) (Snapshot[TKind, TRel, TAttr], error) {
	var empty Snapshot[TKind, TRel, TAttr]
	if maxBytes <= 0 || maxSupports <= 0 ||
		!requiredIdentities(record.Metadata.Run, record.Metadata.ExtractionFingerprint) ||
		(record.Metadata.Parent != "" && !validID(record.Metadata.Parent)) {
		return empty, ragy.ErrInvalidArgument
	}
	supports, err := locations(record, maxSupports)
	if err != nil {
		return empty, err
	}
	if err = authorize(ctx, read, supports, admit); err != nil {
		return empty, err
	}
	data, err := json.Marshal(record)
	if err != nil || len(data) > maxBytes {
		return empty, ragy.ErrInvalidArgument
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	// Reject a BYOT custom marshaler that changes the declared support inventory.
	var owned Record[TKind, TRel, TAttr]
	if err = decode(data, &owned); err != nil {
		return empty, err
	}
	ownedLocations, err := locations(owned, maxSupports)
	if err != nil || !slices.Equal(supports, ownedLocations) {
		return empty, ragy.ErrProtocol
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	digest := sha256.Sum256(data)
	return Snapshot[TKind, TRel, TAttr]{
		data:      data,
		reference: Reference{ID: hex.EncodeToString(digest[:]), Supports: supports},
	}, nil
}

func (s Snapshot[TKind, TRel, TAttr]) Reference() Reference {
	return Reference{ID: s.reference.ID, Supports: slices.Clone(s.reference.Supports)}
}

func (s Snapshot[TKind, TRel, TAttr]) Record() (Record[TKind, TRel, TAttr], error) {
	var result Record[TKind, TRel, TAttr]
	if len(s.data) == 0 {
		return result, ragy.ErrInvalidArgument
	}
	err := decode(s.data, &result)
	return result, err
}

func decode[T any](data []byte, result *T) error {
	if !utf8.Valid(data) {
		return ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(result); err != nil {
		return ragy.ErrProtocol
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return ragy.ErrProtocol
	}
	return nil
}

func validID(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == hex.EncodeToString(decoded)
}

func authorize(ctx context.Context, read access.Binding, supports []source.Locator, admit Admission) error {
	if admit == nil || len(supports) == 0 {
		return ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return err
	}
	for _, location := range supports {
		if err := location.Validate(); err != nil {
			return err
		}
	}
	for _, location := range supports {
		if err := read.Check(ctx); err != nil {
			return err
		}
		if !published(read.Publication(), location.Reference) {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
		if err := admit(ctx, read, location); err != nil {
			return access.NonSkippable(err)
		}
		if err := read.Check(ctx); err != nil {
			return err
		}
	}
	return nil
}

func published(publication access.Publication, reference source.Reference) bool {
	if publication.IsCurrent() {
		return true
	}
	for _, target := range publication.Targets() {
		if target.Namespace == reference.Namespace && target.Source == reference.Source &&
			target.Revision == reference.Revision && target.AccessFingerprint == reference.AccessFingerprint {
			return true
		}
	}
	return false
}

func locations[TKind, TRel comparable, TAttr any](
	record Record[TKind, TRel, TAttr],
	limit int,
) ([]source.Locator, error) {
	if !boundedRecord(record, limit) {
		return nil, ragy.ErrProtocol
	}
	if err := validateRecordIdentities(record); err != nil {
		return nil, err
	}
	if len(record.Input.Entities) != len(record.Result.EntityDecisions) ||
		len(record.Input.Relations) != len(record.Result.RelationDecisions) {
		return nil, ragy.ErrProtocol
	}
	var result []source.Locator
	for index, entity := range record.Input.Entities {
		trace := record.Result.EntityDecisions[index]
		if trace.Mention != entity.ID || !slices.Equal(trace.Supports, entity.Supports) || len(entity.Supports) == 0 {
			return nil, ragy.ErrProtocol
		}
		result = union(result, entity.Supports)
	}
	for index, relation := range record.Input.Relations {
		trace := record.Result.RelationDecisions[index]
		if trace.Mention != relation.ID || !slices.Equal(trace.Supports, relation.Supports) ||
			len(relation.Supports) == 0 {
			return nil, ragy.ErrProtocol
		}
		result = union(result, relation.Supports)
	}
	if err := resultSupports(record.Result, result); err != nil {
		return nil, err
	}
	return result, nil
}

func union(existing, incoming []source.Locator) []source.Locator {
	for _, location := range incoming {
		if !slices.Contains(existing, location) {
			existing = append(existing, location)
		}
	}
	return existing
}

func resultSupports[TKind, TRel comparable, TAttr any](
	result resolution.Result[TKind, TRel, TAttr],
	supports []source.Locator,
) error {
	for _, entity := range result.Entities {
		for _, variant := range entity.Variants {
			if !contained(supports, variant.Supports) {
				return ragy.ErrProtocol
			}
		}
	}
	for _, relation := range result.Relations {
		for _, variant := range relation.Variants {
			if !contained(supports, variant.Supports) {
				return ragy.ErrProtocol
			}
		}
	}
	for _, unresolved := range result.Unresolved {
		if !contained(supports, unresolved.Supports) {
			return ragy.ErrProtocol
		}
	}
	return nil
}

func contained(all, subset []source.Locator) bool {
	if len(subset) == 0 {
		return false
	}
	for _, item := range subset {
		if !slices.Contains(all, item) {
			return false
		}
	}
	return true
}

func boundedRecord[TKind, TRel comparable, TAttr any](record Record[TKind, TRel, TAttr], limit int) bool {
	if limit <= 0 || len(record.Input.Entities) > limit || len(record.Input.Relations) > limit ||
		len(record.Result.Entities) > limit ||
		len(record.Result.Relations) > limit ||
		len(record.Result.Unresolved) > limit {
		return false
	}
	remaining := limit
	for _, entity := range record.Input.Entities {
		if len(entity.Supports) > remaining {
			return false
		}
		remaining -= len(entity.Supports)
	}
	for _, relation := range record.Input.Relations {
		if len(relation.Supports) > remaining {
			return false
		}
		remaining -= len(relation.Supports)
	}
	return boundedResult(record.Result, limit)
}
func boundedResult[TKind, TRel comparable, TAttr any](result resolution.Result[TKind, TRel, TAttr], limit int) bool {
	remaining := limit
	for _, entity := range result.Entities {
		if len(entity.Variants) > limit {
			return false
		}
		for _, variant := range entity.Variants {
			if len(variant.Supports) > remaining {
				return false
			}
			remaining -= len(variant.Supports)
		}
	}
	for _, relation := range result.Relations {
		if len(relation.Variants) > limit {
			return false
		}
		for _, variant := range relation.Variants {
			if len(variant.Supports) > remaining {
				return false
			}
			remaining -= len(variant.Supports)
		}
	}
	for _, unresolved := range result.Unresolved {
		if len(unresolved.Supports) > remaining {
			return false
		}
		remaining -= len(unresolved.Supports)
	}
	return true
}
