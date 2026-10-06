package otel

import (
	"context"
	"reflect"
	"testing"

	"go.opentelemetry.io/otel/trace/noop"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/multimodal"
	"github.com/skosovsky/ragy/tensor"
)

type denseEncodingSpy struct {
	request dense.Request
	result  dense.Result
}

func (s *denseEncodingSpy) Space() dense.Space { return tracedSpace() }
func (s *denseEncodingSpy) Embed(_ context.Context, r dense.Request) (dense.Result, error) {
	s.request = r
	return s.result, nil
}

type tensorEncodingSpy struct {
	request tensor.Request
	result  tensor.Result
}

func (s *tensorEncodingSpy) Space() tensor.Space { return tracedSpace() }
func (s *tensorEncodingSpy) Embed(_ context.Context, r tensor.Request) (tensor.Result, error) {
	s.request = r
	return s.result, nil
}

type multimodalEncodingSpy struct {
	request multimodal.Request
	result  multimodal.Result
}

func (s *multimodalEncodingSpy) Space() dense.Space { return tracedSpace() }
func (s *multimodalEncodingSpy) Embed(_ context.Context, r multimodal.Request) (multimodal.Result, error) {
	s.request = r
	return s.result, nil
}
func TestEmbeddingWrappersPreserveContracts(t *testing.T) {
	usage := embedding.Usage{InputTokens: 12, InputTokensKnown: true}
	tracer := noop.NewTracerProvider().Tracer("contract")
	t.Run("dense", func(t *testing.T) {
		request := dense.Request{Inputs: []string{"query"}, Purpose: embedding.Query, RequireRemoteTokenBound: true}
		spy := &denseEncodingSpy{
			result: dense.Result{
				Embeddings: []dense.Embedding{{Space: tracedSpace(), Vector: []float32{1}}},
				Usage:      usage,
			},
		}
		wrapped, err := WrapDenseEmbedder(spy, tracer)
		if err != nil {
			t.Fatal(err)
		}
		result, err := wrapped.Embed(context.Background(), request)
		if err != nil || !reflect.DeepEqual(spy.request, request) || !reflect.DeepEqual(result, spy.result) ||
			wrapped.Space() != spy.Space() {
			t.Fatalf("roundtrip=%+v,%v", result, err)
		}
	})
	t.Run("tensor", func(t *testing.T) {
		request := tensor.Request{
			Inputs:                  []string{"document"},
			Purpose:                 embedding.Document,
			RequireRemoteTokenBound: true,
		}
		spy := &tensorEncodingSpy{
			result: tensor.Result{
				Embeddings: []tensor.Embedding{{Space: tracedSpace(), Tokens: tensor.Tensor{{1}}}},
				Usage:      usage,
			},
		}
		wrapped, err := WrapTensorEmbedder(spy, tracer)
		if err != nil {
			t.Fatal(err)
		}
		result, err := wrapped.Embed(context.Background(), request)
		if err != nil || !reflect.DeepEqual(spy.request, request) || !reflect.DeepEqual(result, spy.result) ||
			wrapped.Space() != spy.Space() {
			t.Fatalf("roundtrip=%+v,%v", result, err)
		}
	})
	t.Run("multimodal", func(t *testing.T) {
		request := multimodal.Request{
			Inputs: []multimodal.Input{
				{Parts: []multimodal.Part{{Kind: multimodal.PartText, Text: "query"}}},
			},
			Purpose:                 embedding.Query,
			RequireRemoteTokenBound: true,
		}
		spy := &multimodalEncodingSpy{
			result: multimodal.Result{
				Embeddings: []dense.Embedding{{Space: tracedSpace(), Vector: []float32{1}}},
				Usage:      usage,
			},
		}
		wrapped, err := WrapMultimodalEmbedder(spy, tracer)
		if err != nil {
			t.Fatal(err)
		}
		result, err := wrapped.Embed(context.Background(), request)
		if err != nil || !reflect.DeepEqual(spy.request, request) || !reflect.DeepEqual(result, spy.result) ||
			wrapped.Space() != spy.Space() {
			t.Fatalf("roundtrip=%+v,%v", result, err)
		}
	})
}
