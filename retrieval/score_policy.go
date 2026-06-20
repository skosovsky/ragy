package retrieval

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// RankToScoreNormalizer explicitly converts rank-only evidence into a score.
type RankToScoreNormalizer interface {
	NormalizeRank(rank int, total int) (float64, error)
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
	return ragy.ClampScore(1 - float64(rank-1)/float64(total-1)), nil
}

// ApplyRankScorePolicy returns a copy of docs with explicit normalized scores
// for scoreless ranked documents.
func ApplyRankScorePolicy[TMeta any](
	docs []Document[TMeta],
	normalizer RankToScoreNormalizer,
) ([]Document[TMeta], error) {
	if normalizer == nil {
		return nil, fmt.Errorf("%w: rank score normalizer", ragy.ErrInvalidArgument)
	}
	out := copyDocuments(docs)
	total := len(out)
	for i, doc := range out {
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
		doc.Score = ragy.ClampScore(score)
		doc.ScoreState = ScoreNormalized
		doc.Rank = rank
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
	if rs == nil || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	docs, err := ApplyRankScorePolicy(rs.Documents(), normalizer)
	if err != nil {
		return NewResultSet(docs, resolver), err
	}
	return NewResultSet(docs, resolver), nil
}
