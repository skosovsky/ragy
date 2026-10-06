package retrieval

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

// SourceLocations returns an owned union of exact mapping and contributor supports.
// It does not infer a source from ID or metadata or authorize another source.
func (d Document[TMeta]) SourceLocations() []source.Locator {
	return combineLocatorSupports(d.SourceSupports, d.SourceMapping.Supports())
}

func validateDocumentSources[TMeta any](doc Document[TMeta]) error {
	if doc.SourceMapping.Text() != "" {
		if err := doc.SourceMapping.Validate(); err != nil {
			return err
		}
		if doc.SourceMapping.Text() != doc.Content {
			return fmt.Errorf("%w: source mapping does not address document content", ragy.ErrInvalidArgument)
		}
	}
	for _, support := range doc.SourceSupports {
		if err := support.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func groupedSourceMapping[TMeta any](docs []Document[TMeta]) (source.MappedText, error) {
	parts := make([]source.MappedText, 0, len(docs))
	for _, doc := range docs {
		if doc.Content == "" {
			continue
		}
		if doc.SourceMapping.Text() == "" {
			// A missing fragment prevents claiming complete coordinate coverage.
			return source.MappedText{}, nil
		}
		parts = append(parts, doc.SourceMapping)
	}
	if len(parts) == 0 {
		return source.MappedText{}, nil
	}
	return source.JoinMapped("\n\n", parts...)
}
