package filter

import (
	"testing"
)

func TestBuilderBuildsCondition(t *testing.T) {
	t.Parallel()

	builder := NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("String(): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	filterBuilder, err := NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}

	cond, err := Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	if IsEmpty(cond.IR()) {
		t.Fatal("condition is empty")
	}
}

func TestBuilderOrderedComparisons(t *testing.T) {
	t.Parallel()

	builder := NewSchema()
	score, err := builder.Int("score")
	if err != nil {
		t.Fatalf("Int(): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	cases := []struct {
		name  string
		build func(*Builder) *Builder
	}{
		{name: "Gt", build: func(b *Builder) *Builder { return Gt(b, score, 10) }},
		{name: "Gte", build: func(b *Builder) *Builder { return Gte(b, score, 10) }},
		{name: "Lt", build: func(b *Builder) *Builder { return Lt(b, score, 10) }},
		{name: "Lte", build: func(b *Builder) *Builder { return Lte(b, score, 10) }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			filterBuilder, err := NewBuilder(schema)
			if err != nil {
				t.Fatalf("NewBuilder(): %v", err)
			}

			cond, err := tc.build(filterBuilder).Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			if IsEmpty(cond.IR()) {
				t.Fatal("condition is empty")
			}
		})
	}
}

func TestBuilderNot(t *testing.T) {
	t.Parallel()

	builder := NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("String(): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	filterBuilder, err := NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}

	cond, err := Not(Eq(filterBuilder, tenant, "acme")).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	if IsEmpty(cond.IR()) {
		t.Fatal("condition is empty")
	}
}

func TestBuilderOr(t *testing.T) {
	t.Parallel()

	builder := NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("String(): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	left, err := NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(left): %v", err)
	}
	right, err := NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(right): %v", err)
	}

	cond, err := Or(Eq(left, tenant, "acme"), Eq(right, tenant, "beta")).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	if IsEmpty(cond.IR()) {
		t.Fatal("condition is empty")
	}
}
