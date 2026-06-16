package retrieval

import "github.com/skosovsky/ragy/filter"

// MatchDocument evaluates a filter condition against document metadata via codec.
func MatchDocument[TMeta any](
	codec MetadataCodec[TMeta],
	doc Document[TMeta],
	cond filter.Condition,
) (bool, error) {
	attrs, err := codec.Encode(doc.Meta)
	if err != nil {
		return false, err
	}
	return filter.MatchCondition(cond, func(field string) (any, bool) {
		value, ok := attrs[field]
		return value, ok
	})
}
