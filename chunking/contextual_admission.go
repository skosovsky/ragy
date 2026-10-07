package chunking

import (
	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func validateContextualBatch[TMeta any](doc retrieval.Document[TMeta], chunks []Chunk[TMeta]) error {
	for index, chunk := range chunks {
		if ValidateChunk(chunk) != nil || chunk.Index != index || chunk.SourceID != doc.ID ||
			(chunk.Total != 0 && chunk.Total != len(chunks)) {
			return ragy.ErrProtocol
		}
		if chunk.InputSpan != (source.ByteSpan{Start: 0, End: 0}) {
			if chunk.InputSpan.ValidateText(doc.Content) != nil ||
				doc.Content[chunk.InputSpan.Start:chunk.InputSpan.End] != chunk.Content {
				return ragy.ErrProtocol
			}
		}
	}
	return nil
}
