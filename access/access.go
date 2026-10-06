// Package access carries immutable host authorization and publication bindings.
// The host decides authorization; ragy enforces the supplied binding mechanically.
package access

import (
	"context"
	"errors"
	"fmt"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// ProtectionError marks failures that composition must never rescue into success.
// Error text excludes policy values and identifiers; Unwrap preserves classification.
type ProtectionError struct{ Cause error }

func (e *ProtectionError) Error() string { return "read protection failed" }
func (e *ProtectionError) Unwrap() error { return e.Cause }

// Protect classifies an access/publication failure without exposing scope data.
func Protect(err error) error {
	if err == nil {
		return nil
	}
	if protected, ok := errors.AsType[*ProtectionError](err); ok {
		return protected
	}
	return &ProtectionError{Cause: err}
}

// IsProtectionFailure recognizes fatal binding failures through joined errors.
func IsProtectionFailure(err error) bool {
	var protected *ProtectionError
	return errors.As(err, &protected)
}

type unsupportedCapabilityError struct{ cause error }

func (*unsupportedCapabilityError) Error() string     { return "unsupported read capability" }
func (e *unsupportedCapabilityError) Unwrap() []error { return []error{ragy.ErrUnsupported, e.cause} }

// UnsupportedCapability classifies only negotiation failures before target I/O.
// It must never be used to classify authorization failure or a target I/O error.
func UnsupportedCapability(cause error) error {
	return Protect(&unsupportedCapabilityError{cause: cause})
}

// IsUnsupportedCapability distinguishes skippable admission from freshness denial.
// Only a direct capability failure is skippable, not joined/embedded side errors.
//
//nolint:errorlint // Exact admission boundary is required; unwrapping could make authority denial skippable.
func IsUnsupportedCapability(err error) bool {
	protected, ok := err.(*ProtectionError)
	if !ok {
		return false
	}
	_, ok = protected.Cause.(*unsupportedCapabilityError)
	return ok
}

type nonSkippableError struct{ cause error }

func (*nonSkippableError) Error() string   { return "non-skippable read failure" }
func (e *nonSkippableError) Unwrap() error { return e.cause }

// NonSkippable preserves protection classification while forbidding capability
// skipping. Joined protected side errors are sanitized by Protect first.
func NonSkippable(err error) error {
	if err == nil {
		return nil
	}
	return &ProtectionError{Cause: &nonSkippableError{cause: Protect(err)}}
}

// Snapshot identifies the host's authorization decision, not a query hint.
type Snapshot struct {
	Identity    string    `json:"identity"`
	PolicyEpoch int64     `json:"policy_epoch"`
	IssuedAt    time.Time `json:"issued_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

// Authority verifies the same authorization snapshot against current host policy.
// Revocation/unavailability must return an error. It must not grant a new scope.
type Authority interface {
	ValidateRead(context.Context, Snapshot) error
}

// AuthorityFunc adapts a host freshness function.
type AuthorityFunc func(context.Context, Snapshot) error

func (f AuthorityFunc) ValidateRead(ctx context.Context, s Snapshot) error {
	if f == nil {
		return ragy.ErrInvalidArgument
	}
	return f(ctx, s)
}

// TargetRevision pins a managed source revision on one target.
type TargetRevision struct {
	Target            string `json:"target"`
	Namespace         string `json:"namespace"`
	Source            string `json:"source"`
	Revision          string `json:"revision"`
	Transformation    string `json:"transformation"`
	AccessFingerprint string `json:"access_fingerprint"`
}

// Publication is an immutable logical read snapshot. Current is an explicit live
// read profile, without a claim of pinned revision or cross-target consistency.
type Publication struct {
	ref      string
	current  bool
	targets  []TargetRevision
	excluded []string
}

// CurrentPublication chooses live reads explicitly.
func CurrentPublication() Publication {
	return Publication{ref: "current", current: true, targets: nil, excluded: nil}
}

// PinPublication copies a host-provided logical publication inventory.
// Empty inventory is an explicit pinned complete-empty snapshot, never live reads.
func PinPublication(ref string, targets []TargetRevision) (Publication, error) {
	if ref == "" || ref == "current" {
		return Publication{}, Protect(ragy.ErrInvalidArgument)
	}
	type targetKey struct{ target, namespace, source string }
	seen := map[targetKey]struct{}{}
	for _, target := range targets {
		if target.Target == "" || target.Namespace == "" || target.Source == "" || target.Revision == "" ||
			target.Transformation == "" ||
			target.AccessFingerprint == "" {
			return Publication{}, Protect(ragy.ErrInvalidArgument)
		}
		key := targetKey{target: target.Target, namespace: target.Namespace, source: target.Source}
		if _, exists := seen[key]; exists {
			return Publication{}, Protect(ragy.ErrInvalidArgument)
		}
		seen[key] = struct{}{}
	}
	return Publication{ref: ref, current: false, targets: append([]TargetRevision(nil), targets...), excluded: nil}, nil
}

// Reference returns the logical publication identity.
func (p Publication) Reference() string { return p.ref }

// IsCurrent reports the explicitly selected live profile.
func (p Publication) IsCurrent() bool { return p.current }

// Targets returns a defensive inventory copy.
func (p Publication) Targets() []TargetRevision { return append([]TargetRevision(nil), p.targets...) }

// ScopedConfig binds trusted predicates to one host authorization snapshot.
type ScopedConfig struct {
	Snapshot    Snapshot
	Mandatory   filter.Condition
	Schema      filter.Schema
	Publication Publication
	Authority   Authority
	Now         func() time.Time
}

type state struct {
	scoped      bool
	snapshot    Snapshot
	mandatory   filter.Condition
	publication Publication
	authority   Authority
	now         func() time.Time
}

// Binding exposes no setters or mutable collections. Copies retain identity.
// An uninitialized Binding is invalid, never implicitly unrestricted.
type Binding struct{ state *state }

// Unrestricted selects live unrestricted reads explicitly.
func Unrestricted() Binding {
	return Binding{
		state: &state{
			scoped:      false,
			snapshot:    Snapshot{Identity: "", PolicyEpoch: 0, IssuedAt: time.Time{}, ExpiresAt: time.Time{}},
			mandatory:   filter.Condition{},
			publication: CurrentPublication(),
			authority:   nil,
			now:         nil,
		},
	}
}

// UnrestrictedAt chooses an explicit pinned publication without authorization restrictions.
func UnrestrictedAt(publication Publication) (Binding, error) {
	if publication.ref == "" {
		return Binding{}, Protect(ragy.ErrInvalidArgument)
	}
	b := Unrestricted()
	b.state.publication = publication
	return b, nil
}

// Scoped validates and captures immutable host policy before query planning.
func Scoped(config ScopedConfig) (Binding, error) {
	s := config.Snapshot
	if s.Identity == "" || s.PolicyEpoch < 0 || s.IssuedAt.IsZero() || !s.ExpiresAt.After(s.IssuedAt) ||
		config.Authority == nil ||
		config.Now == nil ||
		config.Publication.ref == "" ||
		filter.IsEmpty(config.Mandatory.IR()) {
		return Binding{}, Protect(ragy.ErrInvalidArgument)
	}
	mandatory, err := filter.Intersect(config.Schema, config.Mandatory)
	if err != nil {
		return Binding{}, Protect(err)
	}
	if err := filter.ValidateScopeProfile(mandatory); err != nil {
		return Binding{}, Protect(err)
	}
	return Binding{
		state: &state{
			scoped:      true,
			snapshot:    s,
			mandatory:   mandatory,
			publication: config.Publication,
			authority:   config.Authority,
			now:         config.Now,
		},
	}, nil
}

// Validate rejects missing authorization/publication binding.
func (b Binding) Validate() error {
	if b.state == nil || b.state.publication.ref == "" {
		return Protect(ragy.ErrInvalidArgument)
	}
	return nil
}

// IsScoped reports whether a valid binding restricts access.
func (b Binding) IsScoped() bool { return b.state != nil && b.state.scoped }

// Publication returns the immutable pinned/live read selection.
func (b Binding) Publication() Publication {
	if b.state == nil {
		return Publication{}
	}
	return b.state.publication
}

// Snapshot returns a value-only authorization identity.
func (b Binding) Snapshot() Snapshot {
	if b.state == nil {
		return Snapshot{}
	}
	return b.state.snapshot
}

// Check performs required freshness validation, including revocation before TTL.
// Gate failure is fail-closed and cannot be converted to ordinary rescue.
func (b Binding) Check(ctx context.Context) error {
	if err := b.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return Protect(err)
	}
	if !b.state.scoped {
		return nil
	}
	if !b.state.now().Before(b.state.snapshot.ExpiresAt) {
		return Protect(fmt.Errorf("%w: authorization expired", ragy.ErrUnavailable))
	}
	if err := b.state.authority.ValidateRead(ctx, b.state.snapshot); err != nil {
		return NonSkippable(err)
	}
	if err := ctx.Err(); err != nil {
		return Protect(err)
	}
	return nil
}

// Capabilities declare guarantees, not post-filtering after private payload loading.
// Custom adapters must also satisfy public admission conformance.
type Capabilities struct {
	ScopeProfile             bool
	PinnedPublication        bool
	RequirePinnedPublication bool
}

// Prepare checks freshness, capabilities and target schema before any target I/O.
// The query cannot weaken a mandatory predicate, even when contradictory.
func (b Binding) Prepare(
	ctx context.Context,
	schema filter.Schema,
	query filter.Condition,
	caps Capabilities,
) (filter.Condition, error) {
	if err := b.Check(ctx); err != nil {
		return filter.Condition{}, err
	}
	if b.state.publication.current && caps.RequirePinnedPublication {
		return filter.Condition{}, UnsupportedCapability(ragy.ErrUnsupported)
	}
	if !b.state.publication.current && !caps.PinnedPublication {
		return filter.Condition{}, UnsupportedCapability(ragy.ErrUnsupported)
	}
	if b.state.scoped && !caps.ScopeProfile {
		return filter.Condition{}, UnsupportedCapability(ragy.ErrUnsupported)
	}
	mandatory := filter.Condition{}
	if b.state.scoped {
		mandatory = b.state.mandatory
	}
	if err := filter.ValidateCondition(query); err != nil {
		return filter.Condition{}, Protect(err)
	}
	effective, err := filter.Intersect(schema, mandatory, query)
	if err != nil {
		return filter.Condition{}, UnsupportedCapability(err)
	}
	return effective, nil
}
