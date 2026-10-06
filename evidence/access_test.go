package evidence_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func scoped(t *testing.T, revoked *bool) (access.Binding, filter.Schema) {
	t.Helper()
	base, _, _ := fixture(t)
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot:  access.Snapshot{Identity: "scope7", PolicyEpoch: 7, IssuedAt: now, ExpiresAt: now.Add(time.Minute)},
		Mandatory: mandatory, Schema: schema, Publication: base.Publication(), Now: func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if *revoked {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return read, schema
}

type revokingSink struct {
	revoked *bool
	calls   int
}

func (s *revokingSink) Write(context.Context, evidence.Record) error {
	s.calls++
	*s.revoked = true
	return nil
}

func TestRevocationDuringPolicyAndSinkSuppressesExportDelivery(t *testing.T) {
	// Arrange.
	revoked := false
	read, schema := scoped(t, &revoked)
	_, input, policy := fixture(t)
	input.Schema, input.Codec = schema, retrieval.NewJSONCodec[metadata](schema)
	input.SourceAdmission = func(ctx context.Context, binding access.Binding, _ source.Reference) error { return binding.Check(ctx) }
	policy.AllowIdentifier = func(evidence.IdentifierKind, string) bool { revoked = true; return true }
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("revoked policy exported record", err)
	}
	if _, snapshotErr := record.Snapshot(); snapshotErr == nil {
		t.Fatal("protected capture returned side outputs")
	}
	revoked = false
	_, _, policy = fixture(t)
	sink := &revokingSink{revoked: &revoked}
	executions := 0
	result, err := evidence.Run(
		context.Background(),
		read,
		evidence.RecordingConfig[int]{Mode: evidence.Required, Sink: sink,
			Execute:     func(context.Context) (int, error) { executions++; return 7, nil },
			CloneResult: func(value int) (int, error) { return value, nil },
			Capture: func(ctx context.Context, binding access.Binding, _ int, _ error) (evidence.Record, error) {
				return evidence.Capture(ctx, binding, input, policy)
			},
		},
	)
	if !errors.Is(err, ragy.ErrUnavailable) || result.Result != 0 || result.Receipt.State != "" || executions != 1 ||
		sink.calls != 1 {
		t.Fatal("revoked sink delivery escaped", err)
	}
}

func TestProtectionFailureInCaptureNeverBecomesBestEffortSuccess(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	input.Stages[0].Hits[0].Sources[0].Source = "private-source"
	sink := &spySink{}
	// Act.
	result, err := evidence.Run(
		context.Background(),
		read,
		evidence.RecordingConfig[int]{Mode: evidence.BestEffort, Sink: sink,
			Execute:     func(context.Context) (int, error) { return 7, nil },
			CloneResult: func(value int) (int, error) { return value, nil },
			Capture: func(ctx context.Context, binding access.Binding, _ int, _ error) (evidence.Record, error) {
				return evidence.Capture(ctx, binding, input, policy)
			},
		},
	)
	// Assert.
	if !errors.Is(err, ragy.ErrUnavailable) || result.Result != 0 || result.Receipt.State != "" || sink.calls != 0 {
		t.Fatal("best effort rescued protection failure", err)
	}
}

func TestExportMandatoryMetadataAndSupportAdmissionBeforeIdentifiers(t *testing.T) {
	for _, denied := range []string{"metadata", "support"} {
		t.Run(denied, func(t *testing.T) {
			// Arrange: same published source can still contain scope-denied metadata/support.
			revoked := false
			read, schema := scoped(t, &revoked)
			_, input, policy := fixture(t)
			input.Schema, input.Codec = schema, retrieval.NewJSONCodec[metadata](schema)
			if denied == "metadata" {
				input.Stages[0].Hits[0].Document.Meta.Tenant = "b"
			}
			sourceCalls, hitPolicyCalls := 0, 0
			input.SourceAdmission = func(context.Context, access.Binding, source.Reference) error {
				sourceCalls++
				return ragy.ErrUnavailable
			}
			policy.AllowIdentifier = func(kind evidence.IdentifierKind, _ string) bool {
				if kind == evidence.DocumentIdentifier {
					hitPolicyCalls++
				}
				return true
			}
			// Act.
			record, err := evidence.Capture(context.Background(), read, input, policy)
			// Assert.
			if !errors.Is(err, ragy.ErrUnavailable) || hitPolicyCalls != 0 {
				t.Fatal("scope-denied hit reached identifier policy", err)
			}
			if denied == "metadata" && sourceCalls != 0 {
				t.Fatal("private metadata triggered support lookup")
			}
			if _, recordErr := record.Snapshot(); recordErr == nil {
				t.Fatal("private export returned record")
			}
		})
	}
}
