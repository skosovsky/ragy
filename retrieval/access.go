package retrieval

import (
	"context"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
)

// UnrestrictedRead explicitly chooses the unrestricted live-read profile.
// It does not claim managed publication/snapshot consistency.
func UnrestrictedRead() access.Binding { return access.Unrestricted() }

// ReadCapabilityProvider declares target schema and enforcement guarantees.
// A declaration is a contract, backed by adapter conformance, not a Go sandbox.
type ReadCapabilityProvider interface {
	Schema() filter.Schema
	ReadCapabilities() access.Capabilities
}

// PublicationAdmission explicitly admits a partial pin before target I/O. A host
// implementation must reject excluded targets and preserve incomplete coverage.
type PublicationAdmission interface {
	AdmitPublication(access.Publication) error
}

// PrepareRead checks target admission and mandatory intersection before target I/O.
func PrepareRead[TIntent, TRequestMeta any](
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	provider ReadCapabilityProvider,
) (Request[TIntent, TRequestMeta], error) {
	if isNilReadTarget(provider) {
		return req, access.Protect(ragy.ErrInvalidArgument)
	}
	if err := req.Read.Check(ctx); err != nil {
		return req, err
	}
	if req.Read.Publication().IsPartial() {
		admission, ok := provider.(PublicationAdmission)
		if !ok {
			return req, access.UnsupportedCapability(ragy.ErrUnsupported)
		}
		if err := admission.AdmitPublication(req.Read.Publication()); err != nil {
			return req, err
		}
	}
	query := req.Options.Filters
	if req.Plan != nil {
		combined, err := filter.Intersect(provider.Schema(), query, req.Plan.Filters)
		if err != nil {
			return req, access.UnsupportedCapability(err)
		}
		query = combined
	}
	effective, err := req.Read.Prepare(ctx, provider.Schema(), query, provider.ReadCapabilities())
	if err != nil {
		return req, err
	}
	req.Options.Filters = effective
	return req, nil
}

func admitBackendRead[TIntent, TRequestMeta any](
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	backend any,
) error {
	if err := req.Read.Check(ctx); err != nil {
		return err
	}
	if isNilReadTarget(backend) {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	if req.Read.IsScoped() || !req.Read.Publication().IsCurrent() {
		provider, ok := backend.(ReadCapabilityProvider)
		if !ok {
			return access.UnsupportedCapability(ragy.ErrUnsupported)
		}
		_, err := PrepareRead(ctx, req, provider)
		return err
	}
	return nil
}

func isNilReadTarget(target any) bool {
	if target == nil {
		return true
	}
	value := reflect.ValueOf(target)
	switch value.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
		return value.IsNil()
	case reflect.Invalid, reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128,
		reflect.Array, reflect.String, reflect.Struct, reflect.UnsafePointer:
		return false
	default:
		return false
	}
}

// DeliverRead suppresses every result when protection/freshness fails, including
// partial target returns. It must be called before exposing a payload downstream.
func DeliverRead[TMeta any](
	ctx context.Context,
	binding access.Binding,
	rs ResultSet[TMeta],
	err error,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if readErr := readDeliveryError(ctx, binding, err); readErr != nil {
		return NewResultSet[TMeta](nil, resolver), readErr
	}
	return rs, err
}
func readDeliveryError(ctx context.Context, binding access.Binding, err error) error {
	if gateErr := binding.Check(ctx); gateErr != nil {
		return gateErr
	}
	if access.IsProtectionFailure(err) {
		return access.Protect(err)
	}
	return nil
}

func finishReadResult[TMeta, TExecMeta any](
	ctx context.Context,
	binding access.Binding,
	result RetrievalResult[TMeta, TExecMeta],
	err error,
	resolver IdentityResolver[TMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	if readErr := readDeliveryError(ctx, binding, err); readErr != nil {
		var zero TExecMeta
		return emptyRetrievalResult(resolver, zero), readErr
	}
	result.Coverage = BindPublicationCoverage(binding, result.Coverage)
	return result, err
}
