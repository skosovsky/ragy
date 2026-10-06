package retrieval

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/nilvalue"
)

// RankToScoreNormalizer explicitly converts rank-only evidence into a score.
type RankToScoreNormalizer interface {
	NormalizeRank(rank int, total int) (float64, error)
	Semantics() ScoreSemantics
}

// LinearRankNormalizer maps rank 1..total to 1..0 using a linear scale.
type LinearRankNormalizer struct{}

// NormalizeRank implements RankToScoreNormalizer.
func (LinearRankNormalizer) NormalizeRank(rank int, total int) (float64, error) {
	if rank <= 0 {
		return 0, fmt.Errorf("%w: rank must be > 0", ragy.ErrInvalidArgument)
	}
	if total <= 0 {
		return 0, fmt.Errorf("%w: rank total must be > 0", ragy.ErrInvalidArgument)
	}
	if rank > total {
		return 0, fmt.Errorf("%w: rank cannot exceed total", ragy.ErrInvalidArgument)
	}
	if total == 1 {
		return 1, nil
	}
	return 1 - float64(rank-1)/float64(total-1), nil
}

// ApplyRankScorePolicy returns a copy of docs with explicit normalized scores
// for scoreless ranked documents.
func ApplyRankScorePolicy[TMeta any](
	docs []Document[TMeta],
	normalizer RankToScoreNormalizer,
) ([]Document[TMeta], error) {
	if nilvalue.IsNil(normalizer) {
		return nil, fmt.Errorf("%w: rank score normalizer", ragy.ErrInvalidArgument)
	}
	out := copyDocuments(docs)
	total := len(out)
	for i, doc := range out {
		if err := ValidateDocument(doc); err != nil {
			return out[:i], err
		}
		if doc.ScoreState.IsScored() {
			continue
		}
		rank := doc.Rank
		if rank <= 0 {
			rank = i + 1
		}
		score, err := normalizer.NormalizeRank(rank, total)
		if err != nil {
			return out[:i], err
		}
		doc.Score = score
		doc.ScoreSemantics = normalizer.Semantics()
		doc.ScoreState = ScoreNormalized
		doc.Rank = rank
		if err := ValidateDocument(doc); err != nil {
			return nil, err
		}
		out[i] = doc
	}
	return out, nil
}

// NormalizeRankOnlyResultSet applies an explicit rank-to-score policy to a ResultSet.
func NormalizeRankOnlyResultSet[TMeta any](
	rs ResultSet[TMeta],
	normalizer RankToScoreNormalizer,
) (ResultSet[TMeta], error) {
	resolver := ResolverFor(rs)
	if nilvalue.IsNil(rs) || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	docs, err := ApplyRankScorePolicy(rs.Documents(), normalizer)
	if err != nil {
		return NewResultSet(docs, resolver), err
	}
	return NewResultSet(docs, resolver), nil
}

// Semantics declares the explicitly selected rank normalization policy.
func (LinearRankNormalizer) Semantics() ScoreSemantics { return "rank.linear.relative-batch" }
