package lexical

// SynonymMap expands query tokens with synonym variants.
type SynonymMap map[string][]string

// Expand returns unique tokens including synonyms for each input token.
func (m SynonymMap) Expand(tokens []string) []string {
	if len(tokens) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(tokens))
	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		if _, ok := seen[token]; !ok {
			seen[token] = struct{}{}
			out = append(out, token)
		}
		for _, synonym := range m[token] {
			if synonym == "" {
				continue
			}
			if _, ok := seen[synonym]; ok {
				continue
			}
			seen[synonym] = struct{}{}
			out = append(out, synonym)
		}
	}
	return out
}
