package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/tensor"
)

const sampleCount = 7

type DenseIndexFactory func(t *testing.T) dense.Index
type TensorIndexFactory func(t *testing.T) tensor.Index

// RunDenseIndexSuite checks common dense.Index write semantics.
//
// Adapter-specific payload canonicalization remains covered by backend tests.
// Generic conformance here is intentionally limited to schema exposure and
// write-boundary rejection semantics.
func RunDenseIndexSuite(t *testing.T, factory DenseIndexFactory) {
	t.Helper()

	t.Run("schema exposes declared fields", func(t *testing.T) {
		index := factory(t)
		if _, err := index.Schema().StringField("tenant"); err != nil {
			t.Fatalf("Schema().StringField(tenant): %v", err)
		}
		if _, err := index.Schema().IntField("age"); err != nil {
			t.Fatalf("Schema().IntField(age): %v", err)
		}
	})

	t.Run("invalid attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []dense.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": []string{"x"}},
			Vector:     []float32{1},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(invalid attrs) error = %v, want invalid argument", err)
		}
	})

	t.Run("bad keys reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []dense.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"bad-field": "x"},
			Vector:     []float32{1},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(bad key) error = %v, want invalid argument", err)
		}
	})

	t.Run("unsigned attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []dense.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"age": uint8(sampleCount)},
			Vector:     []float32{1},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(unsigned attr) error = %v, want invalid argument", err)
		}
	})
}

// RunTensorIndexSuite checks common tensor.Index write semantics.
//
// Adapter-specific payload canonicalization remains covered by backend tests.
// Generic conformance here is intentionally limited to schema exposure and
// write-boundary rejection semantics.
func RunTensorIndexSuite(t *testing.T, factory TensorIndexFactory) {
	t.Helper()

	t.Run("schema exposes declared fields", func(t *testing.T) {
		index := factory(t)
		if _, err := index.Schema().StringField("tenant"); err != nil {
			t.Fatalf("Schema().StringField(tenant): %v", err)
		}
	})

	t.Run("invalid attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []tensor.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": []string{"x"}},
			Tensor:     tensor.Tensor{{1}},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(invalid attrs) error = %v, want invalid argument", err)
		}
	})

	t.Run("bad keys reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []tensor.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"bad-field": "x"},
			Tensor:     tensor.Tensor{{1}},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(bad key) error = %v, want invalid argument", err)
		}
	})

	t.Run("unsigned attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []tensor.Record{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": jsonlessUint8(sampleCount)},
			Tensor:     tensor.Tensor{{1}},
		}})
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Upsert(unsigned attr) error = %v, want invalid argument", err)
		}
	})
}

type jsonlessUint8 uint8

func TenantAgeSchema(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(age): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("builder.Build(): %v", err)
	}
	return schema
}

func TenantSchema(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("builder.Build(): %v", err)
	}
	return schema
}
