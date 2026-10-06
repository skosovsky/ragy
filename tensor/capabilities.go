package tensor

// QueryCapabilities declares the exact scoring universe and embedding profile.
// ExactWithinCandidates does not imply global recall or exhaustive index search.
type QueryCapabilities struct {
	Space                 Space
	ScoreSemantics        string
	CandidateLimit        int
	ExactWithinCandidates bool
	Exhaustive            bool
}
