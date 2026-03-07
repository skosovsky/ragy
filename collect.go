package ragy

import "iter"

// Collect consumes the given iter.Seq2 and returns all documents and the first error, if any.
// It stops on the first non-nil error or when the sequence ends.
// Use this when you need a []Document from a lazy Splitter output.
func Collect(seq iter.Seq2[Document, error]) ([]Document, error) {
	var out []Document
	for doc, err := range seq {
		if err != nil {
			return out, err
		}
		out = append(out, doc)
	}
	return out, nil
}
