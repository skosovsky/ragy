package retrieval

import (
	"context"
	"testing"
)

type projectedPlanCaptureBackend struct {
	req Query[struct{}]
}

func (b *projectedPlanCaptureBackend) Retrieve(
	_ context.Context,
	req Query[struct{}],
) (ResultSet[struct{}], error) {
	b.req = req
	return NewResultSet[struct{}](nil, DocumentIDResolver[struct{}]{}), nil
}

func TestProjectedBackendPreservesPlannedQueryWhenProjectorOmitsPlan(t *testing.T) {
	t.Parallel()

	next := &projectedPlanCaptureBackend{}
	backend := ProjectedBackend[intentWithMode, NoRequestMeta, struct{}, NoRequestMeta, struct{}]{
		Next: next,
		Project: func(req Query[intentWithMode]) Query[struct{}] {
			return Query[struct{}]{
				Text:    req.Text,
				Intent:  struct{}{},
				Options: req.Options,
			}
		},
	}

	_, err := backend.Retrieve(context.Background(), Query[intentWithMode]{
		Text: "raw",
		Plan: &PlannedQuery[intentWithMode]{
			Text:         "normalized",
			ExpandedText: "expanded",
			CacheKey:     "cache-key",
		},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if next.req.Plan == nil {
		t.Fatal("projected request Plan = nil, want preserved plan")
	}
	if next.req.EffectiveText() != "expanded" || next.req.Plan.CacheKey != "cache-key" {
		t.Fatalf("projected plan = %#v, want expanded text and cache key preserved", next.req.Plan)
	}
}
