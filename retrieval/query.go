package retrieval

// Query carries retrieval text, host intent, and tuning options.
type Query[TIntent any] struct {
	Text    string
	Intent  TIntent
	Options RetrieveOptions
}
