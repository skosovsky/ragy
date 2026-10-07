// Package bridge demonstrates an optional, host-owned canonical context integration.
package bridge

import (
	"context"
	"errors"

	"github.com/skosovsky/contexty"
	"github.com/skosovsky/memy"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
)

// RankPolicy identifies reciprocal ordinal ranking, ordered by native rank then canonical ID.
const RankPolicy = "host.rank-ordinal/canonical-id-1"

// Reference separates index identity from exact canonical identity.
type Reference struct {
	Scope    memy.Scope   `json:"scope"`
	RecordID string       `json:"record_id"`
	Revision memy.Version `json:"revision"`
}

// Batch carries retrieval coverage without claiming semantic completeness.
type Batch[M any] struct {
	Documents []retrieval.Document[M]
	Coverage  []memy.Coverage
	Partial   bool
	Omissions int
}

// Limits bounds independent final representations; MeasureTokens receives public role/text JSON.
type Limits struct {
	Bytes         int
	Runes         int
	JSONBytes     int
	Tokens        int64
	Tokenizer     string
	MeasureTokens func(context.Context, string) (int64, error)
}

// Options declares host transformation, role and final materialization policy.
// Rewrite is support-only; a nonzero TruncateRunes bounds final text explicitly.
type Options[M any] struct {
	UncertaintyType string
	MessageID       string
	Role            contexty.Role
	Prefix          string
	Suffix          string
	Rewrite         func(context.Context, string) (string, error)
	TruncateRunes   int
	Render          retrieval.ArtifactRenderOptions[M]
	Limits          Limits
}

// Bridge leaves all business types and policies with the host.
// Publish must write synchronously and obey context; use a registered managed sink.
type Bridge[P, R, A, M, U any] struct {
	Engine        *memy.Engine[P, R, A]
	Authority     A
	Scope         memy.Scope
	Purpose       string
	Read          access.Binding
	Retrieve      func(context.Context) (Batch[M], error)
	Map           func(context.Context, retrieval.Document[M]) (Reference, error)
	Project       func(context.Context, memy.Record[P, R]) (retrieval.Document[M], error)
	Uncertainty   func(context.Context, memy.Record[P, R]) (*U, error)
	MaxCandidates int
	Options       Options[M]
	Publish       func(context.Context, Published[U]) error
}

// Published is a detached durable snapshot and its explicit public projection.
type Published[U any] struct {
	Message  contexty.Message
	Durable  []byte
	Public   []byte
	Evidence Sidecar[U]
}

// MappingError keeps private callback values out of public error text.
type MappingError struct {
	Stage   string
	Classes []error
}

func (e *MappingError) Error() string   { return "context bridge: " + e.Stage + " failed" }
func (e *MappingError) Unwrap() []error { return e.Classes }

func failure(stage string, err error) error {
	if err == nil {
		return nil
	}
	classes := []error{}
	for _, class := range []error{context.Canceled, context.DeadlineExceeded, memy.ErrInvalid, memy.ErrUnsupported,
		memy.ErrUnauthorized, memy.ErrScopeViolation, memy.ErrStaleInput, memy.ErrRevoked, memy.ErrNotFound,
		memy.ErrSourceUnavailable, memy.ErrUnavailable, memy.ErrBudget, memy.ErrSchema, memy.ErrConflict,
		memy.ErrInterval, memy.ErrStaleAcceptance, memy.ErrStaleCursor, memy.ErrPolicyDenied,
		memy.ErrMissingEvidence, memy.ErrUnresolvedConflict, memy.ErrIncomparable, memy.ErrVisibilityPending,
		memy.ErrMaintenance, memy.ErrUnknownOutcome, memy.ErrClosed,
		ragy.ErrInvalidArgument, ragy.ErrUnsupported, ragy.ErrProtocol, ragy.ErrUnavailable,
		ragy.ErrMissingID, ragy.ErrMissingSourceID, ragy.ErrEmptyText, ragy.ErrEmptyVector,
		ragy.ErrInvalidPage, ragy.ErrInvalidGraph, retrieval.ErrArtifactLimit} {
		if errors.Is(err, class) {
			classes = append(classes, class)
		}
	}
	return &MappingError{Stage: stage, Classes: classes}
}
