package retrieval

import "github.com/skosovsky/ragy/source"

// evidenceSequence is an operation-scoped deque of borrowed immutable segments.
// Prepending a new winner must preserve the existing evidence order without
// copying its growing prefix. Only snapshot allocates the final owned slice.
type evidenceSequence[T any] struct {
	blocks     []evidenceBlock[T]
	head, tail int
	length     int
}

type evidenceBlock[T any] struct {
	values []T
	next   int
}

func (s *evidenceSequence[T]) add(values []T, front bool) {
	if len(values) == 0 {
		return
	}
	index := len(s.blocks)
	s.blocks = append(s.blocks, evidenceBlock[T]{values: values, next: -1})
	switch {
	case index == 0:
		s.head, s.tail = index, index
	case front:
		s.blocks[index].next = s.head
		s.head = index
	default:
		s.blocks[s.tail].next = index
		s.tail = index
	}
	s.length += len(values)
}

func (s *evidenceSequence[T]) snapshot() []T {
	if s.length == 0 {
		return nil
	}
	out := make([]T, 0, s.length)
	for i := s.head; i != -1; i = s.blocks[i].next {
		out = append(out, s.blocks[i].values...)
	}
	return out
}

type winnerEvidence struct {
	supports evidenceSequence[source.Locator]
	history  evidenceSequence[ScoreObservation]
}

func (e *winnerEvidence) locations() []source.Locator {
	if e.supports.length == 0 {
		return []source.Locator{}
	}
	seen := make(map[source.Locator]struct{}, e.supports.length)
	out := make([]source.Locator, 0, e.supports.length)
	for i := e.supports.head; i != -1; i = e.supports.blocks[i].next {
		for _, location := range e.supports.blocks[i].values {
			if _, exists := seen[location]; exists {
				continue
			}
			seen[location] = struct{}{}
			out = append(out, location)
		}
	}
	return out
}

func observeWinner[T any](e *winnerEvidence, doc Document[T]) {
	if doc.ScoreState.IsScored() {
		e.history.add(
			[]ScoreObservation{
				{DocumentID: doc.ID, Value: doc.Score, State: doc.ScoreState, Semantics: doc.ScoreSemantics},
			},
			false,
		)
	}
}
