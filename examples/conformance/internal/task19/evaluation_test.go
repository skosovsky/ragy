package task19

import (
	"math"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func TestGradedMetricsIndependentExample(t *testing.T) {
	// Arrange: gold grades 3,2,1; a zero-gain item precedes the best source.
	qrels := []Qrel{{"best", 3}, {"mid", 2}, {"low", 1}}
	ids := []string{"irrelevant", "best", "low"}
	// Act.
	recall, mrr, ndcg := Rank(ids, qrels)
	// Assert: hand-calculated DCG 7/log2(3)+1/log2(4), ideal 7+3/log2(3)+1/log2(4).
	want := (7/math.Log2(3) + 0.5) / (7 + 3/math.Log2(3) + 0.5)
	if recall != 2.0/3 || mrr != 0.5 || math.Abs(ndcg-want) > 1e-12 {
		t.Fatalf("%g %g %g", recall, mrr, ndcg)
	}
}
func TestFailuresAndMissingRowsStayInDenominator(t *testing.T) {
	// Arrange.
	split := Split{
		Queries: []Query{
			{ID: "answer", Answerable: true, Qrels: []Qrel{{"gold", 3}}},
			{ID: "absent", Answerable: false},
		},
	}
	rows := []Row{
		{
			Query:      "answer",
			Strategy:   "baseline",
			Repeat:     1,
			Outcome:    "complete",
			Retrieved:  []string{"gold"},
			Delivered:  []string{"gold"},
			UsageKnown: true,
		},
	}
	// Act.
	m := Measure(Corpus{}, split, rows, []string{"baseline"})["baseline"]
	// Assert.
	if m.Answerable != 3 || m.NoAnswer != 3 || m.Failures != 5 || m.Recall != 1.0/3 || m.PolicyKnown {
		t.Fatalf("%+v", m)
	}
}
func TestWrongScopeStaleAndCorruptedCitationFailClosedMetrics(t *testing.T) {
	// Arrange.
	c := Corpus{
		DatasetID: "d",
		Documents: []Document{
			{ID: "private", SourceID: "s", Revision: "r", Scope: "red", Current: true},
			{ID: "stale", SourceID: "s", Revision: "old", Scope: "public", Current: false},
		},
	}
	q := Query{Scope: "public"}
	row := Row{
		Outcome:    "complete",
		UsageKnown: true,
		Retrieved:  []string{"private", "stale"},
		Delivered:  []string{"private"},
		Sources:    []source.Locator{{Reference: source.Reference{Artifact: "private", Source: "fabricated"}}},
	}
	// Act.
	Audit(c, q, &row)
	// Assert.
	if row.ScopeViolations == 0 || row.StaleViolations == 0 || row.CitationViolations == 0 || Compliant(row) {
		t.Fatalf("%+v", row)
	}
}
func TestUnknownAccountingAndOverBudgetCannotQualify(t *testing.T) {
	// Arrange.
	row := Row{Outcome: "complete", UsageKnown: false}
	knownOver := Row{Outcome: "complete", UsageKnown: true, ContextUnits: ContextBytes + 1}
	// Act and Assert.
	if Compliant(row) || Compliant(knownOver) {
		t.Fatal("unknown or overflow qualified")
	}
}

func TestFailedNoAnswerIsNotSuccessfulAbstention(t *testing.T) {
	// Arrange.
	split := Split{Queries: []Query{{ID: "unknown", Answerable: false}}}
	rows := []Row{
		{Query: "unknown", Strategy: "baseline", Repeat: 1, Outcome: "failed"},
		{Query: "unknown", Strategy: "baseline", Repeat: 2, Outcome: "complete", UsageKnown: true},
	}
	// Act.
	got := Measure(Corpus{}, split, rows, []string{"baseline"})["baseline"]
	// Assert: two failures remain in denominator, exactly one observed abstention.
	if got.NoAnswer != 3 || got.Failures != 2 || got.Abstentions != 1 {
		t.Fatalf("%+v", got)
	}
}
func TestUnsupportedDatasetVersionRejected(t *testing.T) {
	// Arrange.
	c := Corpus{SchemaVersion: "2.0.0", DatasetID: "example"}
	s := Split{SchemaVersion: "1.0.0", DatasetID: "example"}
	// Act.
	err := Validate(c, s)
	// Assert.
	if err == nil {
		t.Fatal("unsupported version accepted")
	}
}
func TestCandidateLimitsAndMissingDispatchAccounting(t *testing.T) {
	// Arrange.
	rows := []Row{
		{UsageKnown: true, Outcome: "complete", RetrievalCalls: 1},
		{UsageKnown: true, Outcome: "complete", RetrievalCalls: 1, DispatchCandidates: []int{CandidateLimit + 1}},
	}
	// Act and Assert.
	for _, row := range rows {
		if Compliant(row) {
			t.Fatal("unaccounted or oversized dispatch accepted")
		}
	}
}

func TestFreezeRejectsEmptyManifestAndOperationCounterOverflow(t *testing.T) {
	// Arrange.
	empty := []byte(`{}`)
	row := Row{UsageKnown: true, Outcome: "complete", ModelCalls: ^uint64(0), EncoderCalls: 1}
	// Act.
	err := VerifyFreeze(empty, []byte("corpus"), []byte("split"), "holdout")
	// Assert.
	if err == nil || Compliant(row) {
		t.Fatal("empty freeze or wrapped counter accepted")
	}
}

func TestMalformedSignedAccountingAndDuplicateCandidatesRejected(t *testing.T) {
	// Arrange.
	rows := []Row{
		{UsageKnown: true, Outcome: "complete", ContextUnits: -1},
		{UsageKnown: true, Outcome: "complete", Nanos: -1},
		{UsageKnown: true, Outcome: "complete", ScopeViolations: -1, CitationViolations: 1},
		{UsageKnown: true, Outcome: "complete", CandidateIDs: []string{"same", "same"}},
		{UsageKnown: true, Outcome: "complete", CandidateIDs: []string{""}},
	}
	// Act and Assert.
	for _, row := range rows {
		if Compliant(row) {
			t.Fatal("malformed accounting accepted")
		}
	}
}
