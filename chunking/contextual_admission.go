package chunking

import (
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func validateContextualBatch[TMeta any](doc retrieval.Document[TMeta], chunks []Chunk[TMeta]) error {
	validText := utf8.ValidString(doc.Content)
	for index, chunk := range chunks {
		if ValidateChunk(chunk) != nil || chunk.Index != index || chunk.SourceID != doc.ID ||
			(chunk.Total != 0 && chunk.Total != len(chunks)) {
			return ragy.ErrProtocol
		}
		if chunk.InputSpan != (source.ByteSpan{Start: 0, End: 0}) {
			if !validText || validateSnapshotSpan(chunk.InputSpan, doc.Content) != nil ||
				doc.Content[chunk.InputSpan.Start:chunk.InputSpan.End] != chunk.Content {
				return ragy.ErrProtocol
			}
		}
	}
	return nil
}
