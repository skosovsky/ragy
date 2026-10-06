package source

import (
	"context"
	"fmt"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
)

// LookupRequest carries the trusted read binding and exact requested artifacts.
// References are captured by value before any port is invoked.
type LookupRequest struct {
	Read       access.Binding
	References []Reference
	Filters    filter.Condition
}

// Descriptor contains permission metadata only, never source payload/content.
// Catalog must return an owned metadata snapshot for each exact requested identity.
type Descriptor[TAccess any] struct {
	Reference Reference
	Access    TAccess
}

// Catalog resolves permission metadata for exact retained revisions. Missing/deleted
// revisions return no descriptor or ErrUnavailable; latest substitution is invalid.
type Catalog[TAccess any] interface {
	Describe(context.Context, LookupRequest) ([]Descriptor[TAccess], error)
}

// Materialized associates payload with the identity actually loaded by the host port.
type Materialized[TPayload any] struct {
	Reference Reference
	Payload   TPayload
}

// Loader materializes only the exact references supplied after admission.
// It must preserve revision/representation identity and must not fetch latest on miss.
type Loader[TPayload any] interface {
	Load(context.Context, LookupRequest) ([]Materialized[TPayload], error)
}

// ReadConfig supplies host-owned thin metadata projection and payload snapshots.
// Target is the static publication target identity. Scheduling/retention/auth policy
// remain with the host; this component starts no worker/retry and owns no blob store.
type ReadConfig[TAccess, TPayload any] struct {
	Target          string
	Schema          filter.Schema
	Catalog         Catalog[TAccess]
	Loader          Loader[TPayload]
	Attributes      func(TAccess) (filter.RawAttributes, error)
	ValidatePayload func(Reference, TPayload) error
	ClonePayload    func(TPayload) (TPayload, error)
}

// Reader verifies all identities and scope before any payload loading. Lookup is
// all-or-nothing: missing/denied/malformed entries and gate failure expose no payload.
type Reader[TAccess, TPayload any] struct {
	config ReadConfig[TAccess, TPayload]
}

func NewReader[TAccess, TPayload any](config ReadConfig[TAccess, TPayload]) (*Reader[TAccess, TPayload], error) {
	if config.Target == "" || nilReadPort(config.Catalog) || nilReadPort(config.Loader) ||
		config.Attributes == nil ||
		config.ValidatePayload == nil || config.ClonePayload == nil {
		return nil, fmt.Errorf("%w: source reader configuration", ragy.ErrInvalidArgument)
	}
	if _, err := filter.Intersect(config.Schema); err != nil {
		return nil, err
	}
	return &Reader[TAccess, TPayload]{config: config}, nil
}

// Lookup validates scope/revision before materialization and freshness after I/O
// and metadata projection, including every failure return. No partial payload is
// returned through errors; a host port must not embed payload in its error causes.
func (h *Reader[TAccess, TPayload]) Lookup(
	ctx context.Context,
	request LookupRequest,
) ([]Materialized[TPayload], error) {
	if h == nil {
		return nil, access.NonSkippable(ragy.ErrInvalidArgument)
	}
	captured := request
	captured.References = append([]Reference(nil), request.References...)
	if err := captured.Read.Check(ctx); err != nil {
		return nil, err
	}
	docs, err := h.lookup(ctx, captured)
	if gateErr := captured.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, access.NonSkippable(err)
	}
	return docs, nil
}

func (h *Reader[TAccess, TPayload]) lookup(
	ctx context.Context,
	request LookupRequest,
) ([]Materialized[TPayload], error) {
	if err := validateReferences(request.References); err != nil {
		return nil, err
	}
	if err := validatePublication(request.Read, h.config.Target, request.References); err != nil {
		return nil, err
	}
	effective, err := request.Read.Prepare(
		ctx,
		h.config.Schema,
		request.Filters,
		access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true, PinnedPublication: true},
	)
	if err != nil {
		return nil, err
	}
	request.Filters = effective
	if len(request.References) == 0 {
		return []Materialized[TPayload]{}, nil
	}
	descriptors, err := h.config.Catalog.Describe(ctx, copyLookupRequest(request))
	if gateErr := request.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, err
	}
	if admitErr := h.admitDescriptors(ctx, request, descriptors); admitErr != nil {
		return nil, admitErr
	}
	if gateErr := request.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	payloads, err := h.config.Loader.Load(ctx, copyLookupRequest(request))
	if gateErr := request.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, err
	}
	return h.snapshotPayloads(ctx, request, payloads)
}

func (h *Reader[TAccess, TPayload]) admitDescriptors(
	ctx context.Context,
	request LookupRequest,
	descriptors []Descriptor[TAccess],
) error {
	expected := referenceSet(request.References)
	seen := map[Reference]struct{}{}
	for _, descriptor := range descriptors {
		if _, ok := expected[descriptor.Reference]; !ok {
			return ragy.ErrProtocol
		}
		if _, duplicate := seen[descriptor.Reference]; duplicate {
			return ragy.ErrProtocol
		}
		seen[descriptor.Reference] = struct{}{}
		if err := request.Read.Check(ctx); err != nil {
			return err
		}
		attrs, err := h.config.Attributes(descriptor.Access)
		if err != nil {
			return ragy.WrapProjectionError(err, "source read access")
		}
		normalized, err := h.config.Schema.NormalizeAttributes(attrs)
		if err != nil {
			return ragy.WrapProjectionError(err, "source read access")
		}
		allowed, err := filter.MatchCondition(
			request.Filters,
			func(field string) (any, bool) { value, exists := normalized[field]; return value, exists },
		)
		if err != nil {
			return err
		}
		if !allowed {
			return ragy.ErrUnavailable
		}
	}
	if len(seen) != len(expected) {
		return ragy.ErrUnavailable
	}
	return nil
}

func (h *Reader[TAccess, TPayload]) snapshotPayloads(
	ctx context.Context,
	request LookupRequest,
	payloads []Materialized[TPayload],
) ([]Materialized[TPayload], error) {
	indexed, err := indexPayloads(request.References, payloads)
	if err != nil {
		return nil, err
	}
	// No payload callback runs until every returned identity/count is admitted.
	for _, reference := range request.References {
		if err := request.Read.Check(ctx); err != nil {
			return nil, err
		}
		if err := h.config.ValidatePayload(reference, indexed[reference].Payload); err != nil {
			return nil, err
		}
		if err := request.Read.Check(ctx); err != nil {
			return nil, err
		}
	}
	out := make([]Materialized[TPayload], 0, len(indexed))
	for _, reference := range request.References {
		if err := request.Read.Check(ctx); err != nil {
			return nil, err
		}
		payload := indexed[reference]
		owned, err := h.config.ClonePayload(payload.Payload)
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		if err != nil {
			return nil, err
		}
		if err := h.config.ValidatePayload(reference, owned); err != nil {
			return nil, err
		}
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		payload.Payload = owned
		out = append(out, payload)
	}
	return out, nil
}

func validateReferences(refs []Reference) error {
	seen := map[Reference]struct{}{}
	for _, ref := range refs {
		if err := ref.Validate(); err != nil {
			return err
		}
		if _, duplicate := seen[ref]; duplicate {
			return ragy.ErrInvalidArgument
		}
		seen[ref] = struct{}{}
	}
	return nil
}
func referenceSet(refs []Reference) map[Reference]struct{} {
	out := make(map[Reference]struct{}, len(refs))
	for _, ref := range refs {
		out[ref] = struct{}{}
	}
	return out
}
func copyLookupRequest(request LookupRequest) LookupRequest {
	request.References = append([]Reference(nil), request.References...)
	return request
}
func validatePublication(read access.Binding, target string, refs []Reference) error {
	publication := read.Publication()
	if publication.IsCurrent() {
		return nil
	}
	inventory := publication.Targets()
	for _, ref := range refs {
		found := false
		for _, revision := range inventory {
			if revision.Target == target && revision.Namespace == ref.Namespace && revision.Source == ref.Source &&
				revision.Revision == ref.Revision &&
				revision.Transformation == ref.Transformation &&
				revision.AccessFingerprint == ref.AccessFingerprint {
				found = true
				break
			}
		}
		if !found {
			return ragy.ErrUnavailable
		}
	}
	return nil
}

func nilReadPort(port any) bool {
	if port == nil {
		return true
	}
	value := reflect.ValueOf(port)
	kind := value.Kind()
	if kind == reflect.Pointer || kind == reflect.Interface || kind == reflect.Func || kind == reflect.Map ||
		kind == reflect.Slice ||
		kind == reflect.Chan {
		return value.IsNil()
	}
	return false
}

func indexPayloads[TPayload any](
	references []Reference,
	payloads []Materialized[TPayload],
) (map[Reference]Materialized[TPayload], error) {
	expected := referenceSet(references)
	indexed := make(map[Reference]Materialized[TPayload], len(payloads))
	for _, payload := range payloads {
		if _, exists := expected[payload.Reference]; !exists {
			return nil, ragy.ErrProtocol
		}
		if _, duplicate := indexed[payload.Reference]; duplicate {
			return nil, ragy.ErrProtocol
		}
		indexed[payload.Reference] = payload
	}
	if len(indexed) != len(expected) {
		return nil, ragy.ErrUnavailable
	}
	return indexed, nil
}
