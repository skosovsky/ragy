package contracttest

import (
	"testing"

	"github.com/skosovsky/ragy/retrieval"
)

// ReadDecoratorFixture names a host composition tested with fresh leaf fixtures.
// Decorate must preserve the fixture's I/O observation hooks and binding semantics.
// This is a QA helper, not a production decorator registry.
type ReadDecoratorFixture[TIntent, TRequestMeta, TMeta any] struct {
	Name     string
	Decorate func(*testing.T, retrieval.RequestBackend[TIntent, TRequestMeta, TMeta]) retrieval.RequestBackend[TIntent, TRequestMeta, TMeta]
}

// RunScopedReadDecoratorSuite verifies all direct read cases for each supplied
// composition. A fresh fixture/cache is constructed for every case.
func RunScopedReadDecoratorSuite[TIntent, TRequestMeta, TMeta any](
	t *testing.T,
	factory ScopedReadFactory[TIntent, TRequestMeta, TMeta],
	decorators []ReadDecoratorFixture[TIntent, TRequestMeta, TMeta],
) {
	t.Helper()
	for _, decorator := range decorators {
		t.Run(decorator.Name, func(t *testing.T) {
			RunScopedReadSuite(t, func(t *testing.T) ScopedReadFixture[TIntent, TRequestMeta, TMeta] {
				fixture := factory(t)
				fixture.Backend = decorator.Decorate(t, fixture.Backend)
				return fixture
			})
		})
	}
}
