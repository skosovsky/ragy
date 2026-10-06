//go:build darwin || linux

// Package persistent implements a durable local-filesystem dense target.
package persistent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

const payloadSchema = "ragy.dense-payload/v2"

type Record[TMeta any] struct {
	Reference     source.Reference
	SourceMapping source.MappedText
	Value         dense.Record[TMeta]
}
type Config[TMeta any] struct {
	Root            string
	Namespace       string
	Target          string
	Store           lifecycle.Store
	PayloadReader   lifecycle.PayloadReader
	Schema          filter.Schema
	Codec           retrieval.MetadataCodec[TMeta]
	CloneMeta       func(TMeta) (TMeta, error)
	Space           dense.Space
	MaxCatalogBytes int64
	MaxPayloadBytes int64
	MaxRecords      int
	MaxScanRecords  int
}

type descriptor struct {
	Reference  source.Reference     `json:"reference"`
	Attributes filter.RawAttributes `json:"attributes"`
	Digest     string               `json:"payload_digest"`
}
type catalog struct {
	Schema             string               `json:"schema"`
	Manifest           string               `json:"manifest"`
	Identity           lifecycle.Identity   `json:"identity"`
	PayloadFingerprint string               `json:"payload_fingerprint"`
	Target             string               `json:"target"`
	Space              dense.Space          `json:"space"`
	Records            []descriptor         `json:"records"`
	Artifacts          []lifecycle.Artifact `json:"artifacts"`
}
type payload struct {
	Schema        string            `json:"schema"`
	Reference     source.Reference  `json:"reference"`
	Vector        []float32         `json:"vector"`
	Content       string            `json:"content"`
	SourceMapping source.MappedText `json:"source_mapping"`
}

type Adapter[TMeta any] struct {
	config Config[TMeta]
	root   string
}

func New[TMeta any](config Config[TMeta]) (*Adapter[TMeta], error) {
	if config.Root == "" || config.Namespace == "" || config.Target == "" ||
		config.Store == nil || config.CloneMeta == nil || durablefs.ValidatePayloadReader(config.PayloadReader) != nil ||
		config.MaxRecords <= 0 || config.MaxScanRecords <= 0 || config.MaxCatalogBytes <= 0 || config.MaxPayloadBytes <= 0 ||
		config.MaxCatalogBytes == math.MaxInt64 ||
		config.MaxPayloadBytes == math.MaxInt64 ||
		!utf8.ValidString(config.Namespace) ||
		!utf8.ValidString(config.Target) {
		return nil, ragy.ErrInvalidArgument
	}
	if err := config.Space.Validate(); err != nil {
		return nil, err
	}
	if _, err := filter.Intersect(config.Schema); err != nil {
		return nil, err
	}
	if config.Codec == nil {
		config.Codec = retrieval.NewJSONCodec[TMeta](config.Schema)
	}
	absolute, err := filepath.Abs(config.Root)
	if err != nil {
		return nil, err
	}
	scope, err := json.Marshal(struct {
		Namespace string `json:"namespace"`
		Target    string `json:"target"`
	}{Namespace: config.Namespace, Target: config.Target})
	if err != nil {
		return nil, err
	}
	root := filepath.Join(absolute, digest(scope))
	if err = os.MkdirAll(root, 0o700); err != nil {
		return nil, err
	}
	return &Adapter[TMeta]{config: config, root: root}, nil
}

// Stage validates every record before acquiring the write lock or writing files.
func (a *Adapter[TMeta]) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	records []Record[TMeta],
) (lifecycle.StageResult, error) {
	entry, payloads, err := a.capture(ctx, request, records)
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), true)
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	defer func() { _ = lock.Close() }()
	if err = a.checkStage(ctx, request); err != nil {
		return lifecycle.StageResult{}, err
	}
	existing, err := a.readCatalog(ctx, request.Manifest.ID)
	if err == nil {
		if !equalCatalog(existing, entry) {
			return lifecycle.StageResult{}, lifecycle.ErrConflict
		}
		if err = a.verifyPayloads(ctx, existing); err != nil {
			return lifecycle.StageResult{}, err
		}
		return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: entry.Identity.Revision}, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return lifecycle.StageResult{}, err
	}
	if err = a.install(ctx, request, entry, payloads); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: entry.Identity.Revision}, nil
}
func (a *Adapter[TMeta]) Inspect(ctx context.Context, request lifecycle.StageRequest) (lifecycle.StageResult, error) {
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if request.Manifest.Tombstone {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return lifecycle.StageResult{}, err
	}
	if stageReferences(request) == nil {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), false)
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	defer func() { _ = lock.Close() }()
	entry, err := a.readCatalog(ctx, request.Manifest.ID)
	if errors.Is(err, os.ErrNotExist) {
		return lifecycle.StageResult{State: lifecycle.TargetPending, Revision: ""}, nil
	}
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	if entry.Identity != request.Manifest.Identity || entry.PayloadFingerprint != request.Manifest.Payload ||
		!matchesInventory(entry, request) || !matchesSupports(entry, request) {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	if err = a.verifyPayloads(ctx, entry); err != nil {
		return lifecycle.StageResult{}, err
	}
	if err = ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: entry.Identity.Revision}, nil
}

func (a *Adapter[TMeta]) capture(
	ctx context.Context,
	request lifecycle.StageRequest,
	records []Record[TMeta],
) (catalog, [][]byte, error) {
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace ||
		request.Manifest.Tombstone ||
		len(records) > a.config.MaxRecords {
		return catalog{}, nil, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return catalog{}, nil, err
	}
	planned := stageReferences(request)
	if planned == nil || len(planned) != len(records) {
		return catalog{}, nil, ragy.ErrProtocol
	}
	entry := catalog{
		Schema:             "ragy.dense-index/inventory/v2",
		Manifest:           request.Manifest.ID,
		Identity:           request.Manifest.Identity,
		PayloadFingerprint: request.Manifest.Payload,
		Target:             a.config.Target,
		Space:              a.config.Space,
		Records:            nil,
		Artifacts:          retainedArtifacts(request),
	}
	var payloads [][]byte
	for _, record := range records {
		if err := ctx.Err(); err != nil {
			return catalog{}, nil, err
		}
		if _, exists := planned[record.Reference]; !exists {
			return catalog{}, nil, ragy.ErrProtocol
		}
		delete(planned, record.Reference)
		stored, data, err := a.captureRecord(record)
		if err != nil {
			return catalog{}, nil, err
		}
		entry.Records = append(entry.Records, stored)
		payloads = append(payloads, data)
	}
	// Serialize metadata now to detach arbitrary host codec map ownership.
	encoded, err := json.Marshal(entry)
	if err != nil {
		return catalog{}, nil, err
	}
	if int64(len(encoded)) > a.config.MaxCatalogBytes {
		return catalog{}, nil, ragy.ErrInvalidArgument
	}
	if err = decodeStrict(encoded, &entry); err != nil {
		return catalog{}, nil, err
	}
	return entry, payloads, nil
}
func stageReferences(request lifecycle.StageRequest) map[source.Reference]struct{} {
	for _, target := range request.Manifest.Targets {
		if target.Name != request.Target {
			continue
		}
		refs := make(map[source.Reference]struct{}, len(target.Artifacts))
		for _, artifact := range target.Artifacts {
			refs[artifact.Reference] = struct{}{}
		}
		return refs
	}
	return nil
}
func (a *Adapter[TMeta]) readCatalog(ctx context.Context, id string) (catalog, error) {
	data, err := durablefs.ReadBounded(ctx, filepath.Join(a.path(id), "catalog.json"), a.config.MaxCatalogBytes)
	if err != nil {
		return catalog{}, err
	}
	var entry catalog
	if err = decodeStrict(data, &entry); err != nil {
		return catalog{}, err
	}
	if entry.Schema != "ragy.dense-index/inventory/v2" || entry.Manifest != id || entry.Target != a.config.Target ||
		entry.Identity.Namespace != a.config.Namespace || entry.Space != a.config.Space || len(entry.Records) > a.config.MaxRecords {
		return catalog{}, ragy.ErrProtocol
	}
	if err = entry.Identity.Validate(); err != nil {
		return catalog{}, ragy.ErrProtocol
	}
	if entry.PayloadFingerprint == "" {
		return catalog{}, ragy.ErrProtocol
	}
	inventory, err := validateCatalogArtifacts(entry)
	if err != nil {
		return catalog{}, err
	}
	if err = a.validateCatalogRecords(entry, inventory); err != nil {
		return catalog{}, err
	}
	return entry, nil
}

func (a *Adapter[TMeta]) validateCatalogRecords(
	entry catalog,
	inventory map[source.Reference]struct{},
) error {
	seen := make(map[source.Reference]struct{}, len(entry.Records))
	for i := range entry.Records {
		record := &entry.Records[i]
		if err := record.Reference.Validate(); err != nil {
			return ragy.ErrProtocol
		}
		if _, exists := inventory[record.Reference]; !exists {
			return ragy.ErrProtocol
		}
		if !matchesIdentity(record.Reference, entry.Identity) {
			return ragy.ErrProtocol
		}
		attributes, attrErr := a.config.Schema.NormalizeAttributes(record.Attributes)
		if attrErr != nil {
			return ragy.ErrProtocol
		}
		record.Attributes = attributes
		if _, exists := seen[record.Reference]; exists {
			return ragy.ErrProtocol
		}
		seen[record.Reference] = struct{}{}
		if len(record.Digest) != sha256.Size*2 {
			return ragy.ErrProtocol
		}
		if _, err := hex.DecodeString(record.Digest); err != nil {
			return ragy.ErrProtocol
		}
	}
	return nil
}

func (a *Adapter[TMeta]) verifyPayloads(ctx context.Context, entry catalog) error {
	for _, record := range entry.Records {
		data, err := durablefs.ReadBounded(
			ctx,
			filepath.Join(a.path(entry.Manifest), record.Digest+".json"),
			a.config.MaxPayloadBytes,
		)
		if err != nil {
			return err
		}
		if digest(data) != record.Digest {
			return ragy.ErrProtocol
		}
		var decoded payload
		if err = decodeStrict(data, &decoded); err != nil {
			return err
		}
		if decoded.Schema != payloadSchema || decoded.Reference != record.Reference {
			return ragy.ErrProtocol
		}
		if err = (dense.Embedding{Space: entry.Space, Vector: decoded.Vector}).Validate(); err != nil {
			return ragy.ErrProtocol
		}
	}
	return nil
}
func (a *Adapter[TMeta]) checkStage(ctx context.Context, request lifecycle.StageRequest) error {
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	if !registeredStage(snapshot, request, a.config.Target) {
		return ragy.ErrProtocol
	}
	for _, publication := range snapshot.Publications {
		if publication.Source == request.Manifest.Identity.Source {
			if publication.Manifest != request.Manifest.ExpectedPublication {
				return lifecycle.ErrConflict
			}
			return ctx.Err()
		}
	}
	if request.Manifest.ExpectedPublication != "" {
		return lifecycle.ErrConflict
	}
	return ctx.Err()
}
func (a *Adapter[TMeta]) path(id string) string { return filepath.Join(a.root, digest([]byte(id))) }
func digest(data []byte) string                 { hash := sha256.Sum256(data); return hex.EncodeToString(hash[:]) }
func equalCatalog(first, second catalog) bool {
	a, err := json.Marshal(first)
	if err != nil {
		return false
	}
	b, err := json.Marshal(second)
	return err == nil && bytes.Equal(a, b)
}
func decodeStrict(data []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return ragy.ErrProtocol
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return ragy.ErrProtocol
	}
	return nil
}

func (a *Adapter[TMeta]) install(
	ctx context.Context,
	request lifecycle.StageRequest,
	entry catalog,
	payloads [][]byte,
) error {
	// This reserved path belongs to this exact registered operation. It is
	// never inferred from unknown bootstrap inventory or a source-wide prefix.
	temporary := a.staging(entry.Manifest)
	if err := a.checkStage(ctx, request); err != nil {
		return err
	}
	if err := os.RemoveAll(temporary); err != nil {
		return err
	}
	if err := os.Mkdir(temporary, 0o700); err != nil {
		return err
	}
	var err error
	defer func() { _ = os.RemoveAll(temporary) }()
	for i, record := range entry.Records {
		if err = durablefs.Write(ctx, filepath.Join(temporary, record.Digest+".json"), payloads[i]); err != nil {
			return err
		}
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	if int64(len(data)) > a.config.MaxCatalogBytes {
		return ragy.ErrInvalidArgument
	}
	if err = durablefs.Write(ctx, filepath.Join(temporary, "catalog.json"), data); err != nil {
		return err
	}
	if err = durablefs.SyncDirectory(temporary); err != nil {
		return err
	}
	if err = a.checkStage(ctx, request); err != nil {
		return err
	}
	if err = os.Rename(temporary, a.path(entry.Manifest)); err != nil {
		return err
	}
	if err = durablefs.SyncDirectory(a.root); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	return nil
}

func registeredStage(snapshot lifecycle.Snapshot, request lifecycle.StageRequest, name string) bool {
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != request.Manifest.ID || manifest.Identity != request.Manifest.Identity ||
			manifest.Payload != request.Manifest.Payload || manifest.ExpectedPublication != request.Manifest.ExpectedPublication {
			continue
		}
		for _, target := range manifest.Targets {
			if target.Name == name && target.State == lifecycle.TargetUnknown &&
				lifecycle.SameTargetInventory(manifest, request.Manifest, name) {
				return true
			}
		}
	}
	return false
}
func (a *Adapter[TMeta]) captureRecord(record Record[TMeta]) (descriptor, []byte, error) {
	if err := validateRecordMapping(record.Reference, record.SourceMapping, record.Value.Content); err != nil {
		return descriptor{}, nil, err
	}
	if record.Value.ID != record.Reference.Artifact || record.Value.Space != a.config.Space {
		return descriptor{}, nil, ragy.ErrInvalidArgument
	}
	record.Value.Vector = slices.Clone(record.Value.Vector)
	if err := (dense.Embedding{Space: record.Value.Space, Vector: record.Value.Vector}).Validate(); err != nil {
		return descriptor{}, nil, err
	}
	attrs, err := a.config.Codec.Encode(record.Value.Meta)
	if err != nil {
		return descriptor{}, nil, err
	}
	// Own the transport boundary before any subsequent host codec call.
	attributes, err := json.Marshal(attrs)
	if err != nil {
		return descriptor{}, nil, err
	}
	var owned filter.RawAttributes
	if err = decodeStrict(attributes, &owned); err != nil {
		return descriptor{}, nil, err
	}
	owned, err = a.config.Schema.NormalizeAttributes(owned)
	if err != nil {
		return descriptor{}, nil, err
	}
	data, err := json.Marshal(
		payload{
			Schema:        payloadSchema,
			Reference:     record.Reference,
			Vector:        record.Value.Vector,
			Content:       record.Value.Content,
			SourceMapping: record.SourceMapping,
		},
	)
	if err != nil {
		return descriptor{}, nil, err
	}
	if int64(len(data)) > a.config.MaxPayloadBytes {
		return descriptor{}, nil, ragy.ErrInvalidArgument
	}
	return descriptor{Reference: record.Reference, Attributes: owned, Digest: digest(data)}, data, nil
}

func matchesIdentity(reference source.Reference, identity lifecycle.Identity) bool {
	return reference.Namespace == identity.Namespace && reference.Source == identity.Source &&
		reference.Revision == identity.Revision &&
		reference.Transformation == identity.Transformation &&
		reference.AccessFingerprint == identity.Access
}
func (a *Adapter[TMeta]) staging(id string) string {
	return filepath.Join(a.root, ".stage-"+digest([]byte(id)))
}

func validateRecordMapping(ref source.Reference, mapping source.MappedText, content string) error {
	if mapping.Text() == "" {
		return nil
	}
	if mapping.Validate() != nil || mapping.Text() != content {
		return ragy.ErrInvalidArgument
	}
	for _, location := range mapping.Supports() {
		support := location.Reference
		if support.Namespace != ref.Namespace || support.Source != ref.Source || support.Revision != ref.Revision ||
			support.AccessFingerprint != ref.AccessFingerprint {
			return ragy.ErrInvalidArgument
		}
	}
	return nil
}

// matchesInventory verifies the exact set before any ready outcome. An unchanged
// payload fingerprint alone cannot attest an omitted or substituted artifact.
func matchesInventory(entry catalog, request lifecycle.StageRequest) bool {
	planned := stageReferences(request)
	if planned == nil || len(planned) != len(entry.Records) {
		return false
	}
	for _, record := range entry.Records {
		if _, exists := planned[record.Reference]; !exists {
			return false
		}
		delete(planned, record.Reference)
	}
	return len(planned) == 0
}

func retainedArtifacts(request lifecycle.StageRequest) []lifecycle.Artifact {
	owned := request.Manifest.Clone()
	for _, target := range owned.Targets {
		if target.Name == request.Target {
			return target.Artifacts
		}
	}
	return nil
}
func matchesSupports(entry catalog, request lifecycle.StageRequest) bool {
	for _, target := range request.Manifest.Targets {
		if target.Name == request.Target {
			return lifecycle.SameArtifactInventory(entry.Artifacts, target.Artifacts)
		}
	}
	return false
}

func validateCatalogArtifacts(entry catalog) (map[source.Reference]struct{}, error) {
	if len(entry.Artifacts) != len(entry.Records) {
		return nil, ragy.ErrProtocol
	}
	inventory := make(map[source.Reference]struct{}, len(entry.Artifacts))
	for _, artifact := range entry.Artifacts {
		if artifact.Validate(entry.Identity) != nil {
			return nil, ragy.ErrProtocol
		}
		if _, duplicate := inventory[artifact.Reference]; duplicate {
			return nil, ragy.ErrProtocol
		}
		inventory[artifact.Reference] = struct{}{}
	}
	return inventory, nil
}
