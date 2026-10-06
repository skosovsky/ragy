package otel

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/observation"
)

func TestObserverExportsSafeCompletionAndUnknownUsage(t *testing.T) {
	// Arrange.
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	observer, err := NewObserver(provider.Tracer("safe"))
	if err != nil {
		t.Fatal(err)
	}
	session, err := observation.New(observation.Config{MaxEvents: 32, Observer: observer})
	if err != nil {
		t.Fatal(err)
	}
	ctx := observation.WithQuery(observation.WithSession(context.Background(), session), 0)
	scenarios := []struct {
		stage   observation.Stage
		outcome observation.Outcome
		class   observation.ErrorClass
	}{
		{observation.StageRetrieval, observation.OutcomeSuccess, observation.ErrorNone},
		{observation.StageFallback, observation.OutcomePartial, observation.ErrorUnavailable},
		{observation.StageRescue, observation.OutcomeSuccess, observation.ErrorNone},
		{observation.StageCacheHit, observation.OutcomeSuccess, observation.ErrorNone},
		{observation.StageDelivery, observation.OutcomeExhausted, observation.ErrorResource},
		{observation.StageModel, observation.OutcomeCanceled, observation.ErrorCanceled},
		{observation.StageEncoding, observation.OutcomeUnsupported, observation.ErrorUnsupported},
	}
	// Act.
	for _, scenario := range scenarios {
		_, span := observation.Begin(ctx, scenario.stage)
		span.End(observation.Completion{Outcome: scenario.outcome, Error: scenario.class,
			Usage: observation.Usage{InputTokens: observation.Count{Value: 999}}})
	}
	// Assert.
	spans := exporter.GetSpans()
	if len(spans) != len(scenarios) {
		t.Fatalf("spans=%d", len(spans))
	}
	for i, span := range spans {
		attrs := fmt.Sprint(span.Attributes)
		if span.Name != stageName(scenarios[i].stage) || !strings.Contains(attrs, "ragy.query.ordinal.known") ||
			strings.Contains(attrs, "gen_ai.usage.input_tokens") || strings.Contains(attrs, "999") {
			t.Fatalf("unexpected safe span: %+v", span)
		}
		if scenarios[i].class != observation.ErrorNone && span.Status.Code != codes.Error {
			t.Fatal("missing failure status")
		}
		if span.EndTime.Before(span.StartTime) {
			t.Fatal("negative duration")
		}
	}
	if stats := session.Stats(); stats.Events != uint64(len(scenarios)*2) || stats.Failures != 0 {
		t.Fatalf("stats=%+v", stats)
	}
}

type privacyEncoder struct {
	calls int
	err   error
}

func (*privacyEncoder) Space() dense.Space { return tracedSpace() }
func (e *privacyEncoder) Embed(context.Context, dense.Request) (dense.Result, error) {
	e.calls++
	return dense.Result{Usage: embedding.Usage{InputTokensKnown: true, InputTokens: 12}}, e.err
}
func TestWrapperErrorsNeverExportRawErrorOrInputs(t *testing.T) {
	// Arrange.
	const secret = "Bearer-secret https://private.example/doc#person"
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	encoder := &privacyEncoder{err: fmt.Errorf("%w: %s", ragy.ErrUnavailable, secret)}
	wrapped, err := WrapDenseEmbedder(encoder, provider.Tracer("safe"))
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, gotErr := wrapped.Embed(context.Background(), dense.Request{Inputs: []string{secret}, Purpose: embedding.Query})
	// Assert.
	if !errors.Is(gotErr, ragy.ErrUnavailable) || encoder.calls != 1 {
		t.Fatalf("err=%v calls=%d", gotErr, encoder.calls)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Status.Code != codes.Error || strings.Contains(fmt.Sprint(spans), secret) ||
		len(spans[0].Events) != 0 {
		t.Fatalf("unsafe span: %+v", spans)
	}
	if !strings.Contains(fmt.Sprint(spans[0].Attributes), "gen_ai.usage.input_tokens") {
		t.Fatal("actual usage absent")
	}
}

func TestObserverConcurrentSessionsDoNotCollideOrRetainStarts(t *testing.T) {
	// Arrange.
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	observer, err := NewObserver(provider.Tracer("concurrent"))
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	// Act.
	for range 8 {
		wait.Go(func() {
			session, createErr := observation.New(observation.Config{MaxEvents: 2, Observer: observer})
			if createErr != nil {
				t.Error(createErr)
				return
			}
			_, span := observation.Begin(
				observation.WithSession(context.Background(), session),
				observation.StagePipeline,
			)
			span.End(observation.Completion{Outcome: observation.OutcomeSuccess})
		})
	}
	wait.Wait()
	// Assert.
	if spans := exporter.GetSpans(); len(spans) != 8 {
		t.Fatalf("spans=%d", len(spans))
	}
}

func TestFailedDiagnosticDoesNotRepeatEncodingDispatch(t *testing.T) {
	// Arrange.
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	observer, err := NewObserver(provider.Tracer("failure"))
	if err != nil {
		t.Fatal(err)
	}
	session, err := observation.New(observation.Config{MaxEvents: 2, Observer: observation.ObserverFunc(
		func(ctx context.Context, event observation.Event) error {
			if exportErr := observer.Observe(ctx, event); exportErr != nil {
				return exportErr
			}
			return errors.New("secret diagnostics failure")
		})})
	if err != nil {
		t.Fatal(err)
	}
	encoder := &privacyEncoder{err: nil}
	wrapped, err := WrapDenseEmbedder(encoder, provider.Tracer("wrapper"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, observed := observation.Begin(
		observation.WithSession(context.Background(), session),
		observation.StageEncoding,
	)
	// Act.
	result, err := wrapped.Embed(ctx, dense.Request{Inputs: []string{"secret input"}, Purpose: embedding.Query})
	observed.End(observation.Completion{Outcome: observation.OutcomeSuccess})
	// Assert.
	if err != nil || result.Usage.InputTokens != 12 || encoder.calls != 1 || session.Stats().Failures != 2 {
		t.Fatalf("err=%v calls=%d stats=%+v", err, encoder.calls, session.Stats())
	}
	if strings.Contains(fmt.Sprint(exporter.GetSpans()), "secret") {
		t.Fatal("diagnostics leaked content")
	}
}

func TestWrapperUnknownCountAndPartialFacts(t *testing.T) {
	// Arrange.
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	tracer := provider.Tracer("facts")
	// Act.
	_, failed := tracer.Start(context.Background(), "failed")
	recordOutcome(failed, ragy.ErrUnavailable, 0)
	failed.End()
	_, partial := tracer.Start(context.Background(), "partial")
	recordOutcome(partial, ragy.ErrUnavailable, 2)
	partial.End()
	// Assert.
	spans := exporter.GetSpans()
	for i, span := range spans {
		known := false
		outcome := 0
		for _, attr := range span.Attributes {
			if attr.Key == "ragy.result.count.known" {
				known = attr.Value.AsBool()
			}
			if attr.Key == "ragy.outcome" {
				outcome = int(attr.Value.AsInt64())
			}
		}
		if i == 0 && (known || outcome != int(observation.OutcomeFailed)) {
			t.Fatal("failed dispatch fabricated known zero")
		}
		if i == 1 && (!known || outcome != int(observation.OutcomePartial)) {
			t.Fatal("partial result not observed")
		}
	}
}

func TestStageNamesCoverCurrentContract(t *testing.T) {
	for stage := observation.StagePipeline; stage <= observation.StageLifecycleBootstrap; stage++ {
		if stageName(stage) == "ragy.unknown" {
			t.Fatalf("stage %d unmapped", stage)
		}
	}
}
