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

type DenseIndexFactory func(t *testing.T) dense.Index[StructMeta]
type TensorIndexFactory func(t *testing.T) tensor.Index[StructMeta]

// RunDenseIndexSuite checks common dense.Index write semantics.
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
		err := index.Upsert(context.Background(), []dense.Record[StructMeta]{{
			ID:      "doc-1",
			Content: "hello",
			Meta:    StructMeta{Tenant: "x"},
			Vector:  nil,
		}})
		if err == nil {
			t.Fatal("Upsert(empty vector) error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrEmptyVector) {
			t.Fatalf("Upsert(empty vector) error = %v, want empty vector", err)
		}
	})

	t.Run("bad keys reject on write", func(t *testing.T) {
		index := factory(t)
		_, err := index.Schema().StringField("bad-field")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(bad-field) error = %v, want invalid argument", err)
		}
	})

	t.Run("unsigned attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []dense.Record[StructMeta]{{
			ID:      "",
			Content: "hello",
			Meta:    StructMeta{Age: sampleCount},
			Vector:  []float32{1},
		}})
		if !errors.Is(err, ragy.ErrMissingID) {
			t.Fatalf("Upsert(missing id) error = %v, want missing id", err)
		}
	})
}

// RunTensorIndexSuite checks common tensor.Index write semantics.
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
		err := index.Upsert(context.Background(), []tensor.Record[StructMeta]{{
			ID:      "doc-1",
			Content: "hello",
			Meta:    StructMeta{Tenant: "x"},
			Tensor:  nil,
		}})
		if err == nil {
			t.Fatal("Upsert(empty tensor) error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrEmptyVector) {
			t.Fatalf("Upsert(empty tensor) error = %v, want empty vector", err)
		}
	})

	t.Run("bad keys reject on write", func(t *testing.T) {
		index := factory(t)
		_, err := index.Schema().StringField("bad-field")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(bad-field) error = %v, want invalid argument", err)
		}
	})

	t.Run("unsigned attrs reject on write", func(t *testing.T) {
		index := factory(t)
		err := index.Upsert(context.Background(), []tensor.Record[StructMeta]{{
			ID:      "",
			Content: "hello",
			Meta:    StructMeta{Tenant: "x"},
			Tensor:  tensor.Tensor{{1}},
		}})
		if !errors.Is(err, ragy.ErrMissingID) {
			t.Fatalf("Upsert(missing id) error = %v, want missing id", err)
		}
	})
}

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
