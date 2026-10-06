package graphsummary_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/source"
)

func TestSummaryBudgetStopsBoundedPartialWithoutReduce(t *testing.T) {
	for _, calls := range []uint64{0, 1, 2} {
		t.Run(map[uint64]string{0: "zero", 1: "one-map", 2: "two-maps"}[calls], func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			f.limits.Limits.ModelCalls = calls
			// Act.
			result, ledger, err := run(context.Background(), t, f, true)
			// Assert.
			outcome := recipe.Partial
			if calls == 0 {
				outcome = recipe.Insufficient
			}
			if err != nil || result.Outcome != outcome || result.Stop != graphsummary.BudgetExhausted ||
				result.Global != nil ||
				result.ModelCalls != calls ||
				len(result.Communities) != int(calls) ||
				ledger.Snapshot().Occupied.ModelCalls != calls {
				t.Fatal(result, err, ledger.Snapshot())
			}
		})
	}
}

func TestSummaryWholeBatchAdmissionBeforeModelAndSourceCallbacks(t *testing.T) {
	for _, scenario := range []string{"private", "foreign-revision", "invalid-members", "too-many-snippets", "input-bytes", "canceled"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			configureAdmission(t, f, scenario, cancel)
			// Act.
			result, ledger, err := run(ctx, t, f, true)
			// Assert: invalid/private final community cannot follow source/model dispatch for first one.
			if err == nil || f.calls != 0 || f.sourceCalls != 0 || f.membershipCalls != 0 ||
				len(result.Communities) != 0 ||
				ledger.Snapshot().Occupied.ModelCalls != 0 {
				t.Fatal(result, err, f.calls, f.sourceCalls, f.membershipCalls, ledger.Snapshot())
			}
		})
	}
}

func configureAdmission(t *testing.T, f *fixture, scenario string, cancel context.CancelFunc) {
	t.Helper()
	switch scenario {
	case "private":
		f.request.Communities[1].Snippets[0].Access.Tenant = "b"
	case "foreign-revision":
		snippet := f.request.Communities[1].Snippets[0]
		location := snippet.Mapping.Supports()[0]
		location.Reference.Revision = "r9"
		mapping, err := source.OriginalText(location, snippet.Mapping.Text())
		if err != nil {
			t.Fatal(err)
		}
		f.request.Communities[1].Snippets[0].Mapping = mapping
	case "invalid-members":
		f.request.Communities[1].Snippets[0].Members = []string{"outside-community"}
	case "too-many-snippets":
		for range 20 {
			f.request.Communities[1].Snippets = append(
				f.request.Communities[1].Snippets,
				f.request.Communities[1].Snippets[0],
			)
		}
	case "input-bytes":
		f.config.MaxInputBytes = 4
		f.config.MaxSummaryBytes = 2
	case "canceled":
		cancel()
	}
}

func TestSummaryModelFailuresNeverRetryOrDeliverUnsupportedEvidence(t *testing.T) {
	for _, scenario := range []string{"foreign-index", "duplicate-index", "missing-global-community", "large-text", "output-overrun", "advisory-overrun", "model-error", "deleted-after-map", "revoked-model", "fake-deadline"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			configureModelFailure(f, scenario)
			// Act.
			result, ledger, err := run(context.Background(), t, f, true)
			// Assert.
			want := uint64(1)
			if scenario == "missing-global-community" {
				want = 3
			}
			if err == nil || len(result.Communities) != 0 || result.Global != nil ||
				ledger.Snapshot().Occupied.ModelCalls != want ||
				uint64(f.calls) != want {
				t.Fatal(result, err, f.calls, ledger.Snapshot())
			}
			if (scenario == "output-overrun" || scenario == "advisory-overrun") &&
				!errors.Is(err, budget.ErrUsageExceeded) {
				t.Fatal(err)
			}
		})
	}
}

func configureModelFailure(f *fixture, scenario string) {
	original := f.config.Model
	if scenario == "advisory-overrun" {
		f.limits.RequireKnownCost = false
		f.config.Quote = func(context.Context, graphsummary.Stage) (budget.Reservation, error) {
			return budget.Reservation{
				Kind:      budget.Model,
				Usage:     budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 0},
				CostKnown: false,
			}, nil
		}
	}
	f.config.Model = func(ctx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		output, usage, err := original(ctx, input)
		switch scenario {
		case "foreign-index":
			output.Selected = []int{100}
		case "duplicate-index":
			output.Selected = []int{0, 0}
		case "missing-global-community":
			if input.Stage == graphsummary.Reduce {
				output.Selected = []int{0}
			}
		case "large-text":
			output.Text = string(make([]byte, f.config.MaxSummaryBytes+1))
		case "output-overrun", "advisory-overrun":
			usage.Value.OutputTokens = 257
		case "model-error":
			err = ragy.ErrProtocol
		case "deleted-after-map":
			f.deleted = true
		case "revoked-model":
			f.epoch++
		case "fake-deadline":
			f.now = f.now.Add(6 * time.Second)
		}
		return output, usage, err
	}
}

func TestSummaryInvalidatesDeletionRevocationExpiryAndPublicationChange(t *testing.T) {
	for _, scenario := range []string{"deleted", "revoked", "expired", "publication"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: cache the immutable derived artifact from a successful attempt.
			f := newFixture(t)
			result, _, err := run(context.Background(), t, f, true)
			if err != nil {
				t.Fatal(err)
			}
			read := f.request.Read
			switch scenario {
			case "deleted":
				f.deleted = true
			case "revoked":
				f.epoch++
			case "expired":
				f.now = f.now.Add(31 * time.Second)
			case "publication":
				pub, pubErr := access.PinPublication("pub2", read.Publication().Targets())
				if pubErr != nil {
					t.Fatal(pubErr)
				}
				read, pubErr = access.UnrestrictedAt(pub)
				if pubErr != nil {
					t.Fatal(pubErr)
				}
			}
			// Act.
			mapping, err := result.Global.Resolve(context.Background(), read, f.admitSource)
			// Assert.
			if !access.IsProtectionFailure(err) || mapping.Text() != "" {
				t.Fatal(mapping.Text(), err)
			}
		})
	}
}

func TestSummarySourceDeletionDuringPricingOrCountingPreventsDispatch(t *testing.T) {
	for _, stage := range []string{"pricing", "counting", "partial-stop"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			if stage == "counting" {
				f.config.CountInputTokens = func(graphsummary.ModelInput) (uint64, error) { f.deleted = true; return 32, nil }
			} else {
				quote := f.config.Quote
				f.config.Quote = func(ctx context.Context, input graphsummary.Stage) (budget.Reservation, error) {
					if stage == "pricing" || f.calls == 1 {
						f.deleted = true
					}
					return quote(ctx, input)
				}
				if stage == "partial-stop" {
					f.limits.Limits.ModelCalls = 1
				}
			}
			// Act.
			result, ledger, err := run(context.Background(), t, f, true)
			// Assert.
			want := 0
			if stage == "partial-stop" {
				want = 1
			}
			if !access.IsProtectionFailure(err) || len(result.Communities) != 0 || result.Global != nil ||
				f.calls != want ||
				ledger.Snapshot().Occupied.ModelCalls != uint64(want) {
				t.Fatal(result, err, f.calls, ledger.Snapshot())
			}
		})
	}
}

func TestSummaryInsufficientMembershipCoverageRemainsExplicit(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	f.request.Communities[0].Snippets[0].Members = []string{"Billing", "LedgerDB"}
	// Act.
	result, _, err := run(context.Background(), t, f, true)
	// Assert.
	if err != nil || result.Outcome != recipe.Insufficient || result.Stop != graphsummary.MissingCoverage ||
		result.ModelCalls != 1 ||
		result.Global != nil ||
		len(result.Communities) != 1 ||
		result.Communities[0].CoversMembership() {
		t.Fatal(result, err)
	}
}
