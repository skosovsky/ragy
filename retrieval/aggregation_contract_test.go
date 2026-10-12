package retrieval

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

type aggregationMeta struct {
	Key  string
	Tags []string
}
type aggregationResolver struct{ calls []string }

func (r *aggregationResolver) Resolve(doc Document[aggregationMeta]) Identity {
	r.calls = append(r.calls, doc.ID)
	return Identity{DocumentID: doc.ID, MergeKey: doc.Meta.Key}
}
func aggregationLocation(id string) source.Locator {
	return source.Locator{
		Kind: source.DocumentLocation,
		Reference: source.Reference{
			Namespace:         "test",
			Source:            id,
			Revision:          "r",
			Transformation:    "t",
			AccessFingerprint: "a",
			Artifact:          "artifact",
			Representation:    "text",
		},
	}
}
func aggregationDocument(t *testing.T, index int, score float64, mapped bool) Document[aggregationMeta] {
	t.Helper()
	location := aggregationLocation(strconv.Itoa(index))
	doc := Document[aggregationMeta]{
		ID:             fmt.Sprintf("doc-%d", index),
		Content:        "payload",
		Meta:           aggregationMeta{Key: "shared", Tags: []string{"host-owned"}},
		Score:          score,
		ScoreState:     ScorePresent,
		ScoreSemantics: "native",
		SourceSupports: []source.Locator{location, location},
		ScoreHistory: []ScoreObservation{
			{
				DocumentID: fmt.Sprintf("prior-%d", index),
				Value:      float64(index),
				State:      ScorePresent,
				Semantics:  "prior",
			},
		},
	}
	if mapped {
		location.Kind = source.TextLocation
		location.Span = source.ByteSpan{Start: 0, End: len(doc.Content)}
		var err error
		doc.SourceMapping, err = source.OriginalText(location, doc.Content)
		if err != nil {
			t.Fatal(err)
		}
	}
	return doc
}

// referenceAggregation intentionally uses the original eager keepWinner algorithm,
// with its own ordering, union and score observations. It must not use the new
// accumulator or its helpers: winner replacement ordering is observable evidence.
//
//nolint:gocognit // Keep the full eager baseline oracle independent of optimized production helpers.
func referenceAggregation(
	docs []Document[aggregationMeta],
	operation string,
) ([]Document[aggregationMeta], []string, error) {
	byKey := map[string]Document[aggregationMeta]{}
	var calls []string
	finish := func() []Document[aggregationMeta] {
		keys := make([]string, 0, len(byKey))
		for key := range byKey {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var out []Document[aggregationMeta]
		for _, key := range keys {
			out = append(out, byKey[key])
		}
		sort.SliceStable(out, func(i, j int) bool { return referenceAggregationLess(out[i], out[j]) })
		// Match public snapshots, including nil history and non-nil empty supports.
		for i := range out {
			out[i].SourceSupports = append(out[i].SourceSupports[:0:0], out[i].SourceSupports...)
			out[i].ScoreHistory = append([]ScoreObservation(nil), out[i].ScoreHistory...)
		}
		return out
	}
	var first *Document[aggregationMeta]
	for i, doc := range docs {
		if err := ValidateDocument(doc); err != nil {
			return finish(), calls, ragy.WrapProjectionError(err, operation)
		}
		calls = append(calls, doc.ID)
		key := doc.Meta.Key
		if key == "" {
			return finish(), calls, fmt.Errorf("%w: empty merge key for document %q", ragy.ErrInvalidArgument, doc.ID)
		}
		if first != nil && (first.ScoreState != doc.ScoreState || first.ScoreSemantics != doc.ScoreSemantics) {
			return finish(), calls, fmt.Errorf(
				"%w: incompatible score scales; use explicit rank fusion or normalization",
				ragy.ErrInvalidArgument,
			)
		}
		if first == nil {
			first = &docs[i]
		}
		current, exists := byKey[key]
		if !exists {
			byKey[key] = doc
			continue
		}
		if doc.Content != current.Content || !reflect.DeepEqual(doc.Meta, current.Meta) {
			if referenceAggregationLess(doc, current) {
				byKey[key] = doc
			}
			continue
		}
		winner, loser := current, doc
		if referenceAggregationLess(doc, current) {
			winner, loser = doc, current
		}
		winner.SourceSupports = referenceAggregationUnion(
			referenceAggregationLocations(winner),
			referenceAggregationLocations(loser),
		)
		history := append([]ScoreObservation(nil), winner.ScoreHistory...)
		history = append(history, loser.ScoreHistory...)
		if loser.ScoreState.IsScored() {
			history = append(
				history,
				ScoreObservation{
					DocumentID: loser.ID,
					Value:      loser.Score,
					State:      loser.ScoreState,
					Semantics:  loser.ScoreSemantics,
				},
			)
		}
		winner.ScoreHistory = history
		byKey[key] = winner
	}
	return finish(), calls, nil
}
func referenceAggregationLess(a, b Document[aggregationMeta]) bool {
	if a.ScoreState.IsScored() && b.ScoreState.IsScored() {
		return a.Score > b.Score
	}
	if a.ScoreState.IsScored() {
		return true
	}
	if b.ScoreState.IsScored() {
		return false
	}
	if a.Rank > 0 && b.Rank > 0 {
		return a.Rank < b.Rank
	}
	return a.Rank > 0 && b.Rank == 0
}
func referenceAggregationUnion(parts ...[]source.Locator) []source.Locator {
	out := make([]source.Locator, 0)
	for _, part := range parts {
		for _, location := range part {
			if !slices.Contains(out, location) {
				out = append(out, location)
			}
		}
	}
	return out
}
func referenceAggregationLocations(doc Document[aggregationMeta]) []source.Locator {
	return referenceAggregationUnion(doc.SourceSupports, doc.SourceMapping.Supports())
}

//nolint:gocognit // Exercise both public operations and every Merge boundary against the independent oracle.
func assertAggregationReference(t *testing.T, docs []Document[aggregationMeta]) {
	t.Helper()
	for _, operation := range []string{"dedup", "merge"} {
		t.Run(operation, func(t *testing.T) {
			// Arrange: split Merge at every boundary, exercising left/right snapshots.
			boundaries := []int{0}
			if operation == "merge" {
				boundaries = make([]int, len(docs)+1)
				for i := range boundaries {
					boundaries[i] = i
				}
			}
			for _, boundary := range boundaries {
				resolver := &aggregationResolver{}
				var result ResultSet[aggregationMeta]
				var err error
				var input ResultSet[aggregationMeta]
				if operation == "dedup" {
					input = NewResultSet(docs, resolver)
				} else {
					input = NewResultSet(docs[:boundary], resolver)
				}
				expected, expectedCalls, expectedErr := referenceAggregation(
					NewResultSet(docs, nil).Documents(),
					operation+" validate",
				)
				// Act.
				if operation == "dedup" {
					result, err = input.Dedup()
				} else {
					result, err = input.Merge(NewResultSet(docs[boundary:], nil))
				}
				// Assert: exact values, order, nilness, partial output and callbacks.
				if (err == nil) != (expectedErr == nil) || (err != nil && err.Error() != expectedErr.Error()) {
					t.Fatalf("boundary=%d errors: got %v want %v", boundary, err, expectedErr)
				}
				if expectedErr != nil && errors.Is(expectedErr, ragy.ErrInvalidArgument) &&
					!errors.Is(err, ragy.ErrInvalidArgument) {
					t.Fatalf("error lost sentinel: %v", err)
				}
				if !reflect.DeepEqual(result.Documents(), expected) {
					t.Fatalf("boundary=%d result:\n got %#v\nwant %#v", boundary, result.Documents(), expected)
				}
				if !reflect.DeepEqual(resolver.calls, expectedCalls) {
					t.Fatalf("callbacks got %v want %v", resolver.calls, expectedCalls)
				}
				if ResolverFor(result) != resolver {
					t.Fatal("custom resolver was not retained")
				}
			}
		})
	}
}

//nolint:gocognit // Explicit cross-product preserves evidence, ordering and rank/scored coverage in one matrix.
func TestAggregationReferenceMatrix(t *testing.T) {
	for _, order := range []string{"ascending", "descending", "ties", "alternating"} {
		for _, mapped := range []bool{false, true} {
			for _, rankOnly := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/mapped=%t/rankOnly=%t", order, mapped, rankOnly), func(t *testing.T) {
					// Arrange.
					docs := make([]Document[aggregationMeta], 8)
					for i := range docs {
						score := float64(i)
						switch order {
						case "descending":
							score = float64(8 - i)
						case "ties":
							score = 1
						case "alternating":
							score = float64((i * 3) % 5)
						}
						docs[i] = aggregationDocument(t, i, score, mapped)
						if i%3 == 0 {
							docs[i].SourceSupports = append(docs[i].SourceSupports, aggregationLocation("common"))
						}
						if rankOnly {
							docs[i].Score = 0
							docs[i].ScoreState = ScoreAbsent
							docs[i].ScoreSemantics = ""
							docs[i].Rank = int(score) + 1
						}
					}
					// Act/Assert.
					assertAggregationReference(t, docs)
				})
			}
		}
	}
	t.Run("payload resets and losing payload cannot contribute", func(t *testing.T) {
		// Arrange: replacement resets evidence, subsequent loser cannot contaminate it.
		docs := make([]Document[aggregationMeta], 7)
		for i := range docs {
			docs[i] = aggregationDocument(t, i, float64(i), true)
		}
		docs[2].Content = "different"
		docs[2].SourceMapping = source.MappedText{}
		docs[3].Content = "different"
		docs[3].SourceMapping = source.MappedText{}
		docs[4].Score = 0 // old payload loses against the replacement.
		docs[5].Meta.Tags = []string{"different-metadata"}
		docs[6].Meta.Tags = []string{"different-metadata"}
		// Act/Assert.
		assertAggregationReference(t, docs)
	})
	t.Run("empty and nil slices", func(t *testing.T) {
		assertAggregationReference(t, nil)
		docs := []Document[aggregationMeta]{aggregationDocument(t, 0, 1, false), aggregationDocument(t, 1, 2, false)}
		for i := range docs {
			docs[i].ScoreHistory = nil
			docs[i].SourceSupports = nil
		}
		assertAggregationReference(t, docs)
		for i := range docs {
			docs[i].SourceSupports = []source.Locator{}
			docs[i].ScoreState = ScoreAbsent
			docs[i].Score = 0
			docs[i].ScoreSemantics = ""
		}
		assertAggregationReference(t, docs)
	})
}

func TestAggregationInvalidPartialOutput(t *testing.T) {
	mutations := map[string]func(*Document[aggregationMeta]){
		"invalid locator":         func(d *Document[aggregationMeta]) { d.SourceSupports[0].Reference.Revision = "" },
		"invalid history":         func(d *Document[aggregationMeta]) { d.ScoreHistory[0].Value = math.Inf(1) },
		"mapping mismatch":        func(d *Document[aggregationMeta]) { d.Content = "wrong mapping" },
		"mixed scale":             func(d *Document[aggregationMeta]) { d.ScoreSemantics = "other" },
		"mixed scored rank-only":  func(d *Document[aggregationMeta]) { d.ScoreState = ScoreAbsent; d.Score = 0; d.ScoreSemantics = "" },
		"mixed normalized native": func(d *Document[aggregationMeta]) { d.ScoreState = ScoreNormalized; d.Score = 0.5 },
		"missing id":              func(d *Document[aggregationMeta]) { d.ID = "" },
		"empty merge key":         func(d *Document[aggregationMeta]) { d.Meta.Key = "" },
		"negative rank":           func(d *Document[aggregationMeta]) { d.Rank = -1 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			// Arrange: completed duplicate accumulation before an invalid contributor.
			docs := []Document[aggregationMeta]{
				aggregationDocument(t, 0, 1, true),
				aggregationDocument(t, 1, 2, true),
				aggregationDocument(t, 2, 3, true),
				aggregationDocument(t, 3, 4, true),
			}
			mutate(&docs[2])
			// Act/Assert.
			assertAggregationReference(t, docs)
		})
	}
}

func TestAggregationDeterministicDifferential(t *testing.T) {
	// Arrange: seeded sequences include multiple keys, payload resets, repeated
	// supports, mappings, winner ties and scored/rank-only ordering.
	random := rand.New(rand.NewSource(20261012))
	for caseIndex := range 128 {
		t.Run(strconv.Itoa(caseIndex), func(t *testing.T) {
			docs := make([]Document[aggregationMeta], random.Intn(9)+1)
			for i := range docs {
				docs[i] = aggregationDocument(t, i, float64(random.Intn(4)), random.Intn(2) == 0)
				docs[i].Meta.Key = strconv.Itoa(random.Intn(3))
				docs[i].Meta.Tags = []string{strconv.Itoa(random.Intn(2))}
				if random.Intn(2) == 0 {
					docs[i].Content = "other"
					docs[i].SourceMapping = source.MappedText{}
				}
				if random.Intn(3) == 0 {
					docs[i].SourceSupports = nil
				}
				if random.Intn(3) == 0 {
					docs[i].ScoreHistory = nil
				}
				if caseIndex%2 == 0 {
					docs[i].ScoreState = ScoreAbsent
					docs[i].Score = 0
					docs[i].ScoreSemantics = ""
					docs[i].Rank = random.Intn(4)
				}
			}
			// Act/Assert.
			assertAggregationReference(t, docs)
		})
	}
}

func TestAggregationOwnedEvidenceAndBYOT(t *testing.T) {
	for _, operation := range []string{"dedup", "merge"} {
		t.Run(operation, func(t *testing.T) {
			// Arrange.
			docs := []Document[aggregationMeta]{aggregationDocument(t, 0, 1, true), aggregationDocument(t, 1, 2, true)}
			resolver := &aggregationResolver{}
			left := NewResultSet(docs[:1], resolver)
			right := NewResultSet(docs[1:], nil)
			input := NewResultSet(docs, resolver)
			original := input.Documents()
			// Act.
			var result ResultSet[aggregationMeta]
			var err error
			if operation == "dedup" {
				result, err = input.Dedup()
			} else {
				result, err = left.Merge(right)
			}
			if err != nil {
				t.Fatal(err)
			}
			expected := result.Documents()
			docs[0].SourceSupports[0].Reference.Source = "input-mutated"
			docs[0].ScoreHistory[0].DocumentID = "input-mutated"
			output := result.Documents()
			output[0].SourceSupports[0].Reference.Source = "output-mutated"
			output[0].ScoreHistory[0].DocumentID = "output-mutated"
			fragments := output[0].SourceMapping.Fragments()
			fragments[0].Supports[0].Reference.Source = "mapping-mutated"
			// Assert: all ragy evidence stays isolated; arbitrary BYOT metadata stays host-owned.
			if !reflect.DeepEqual(result.Documents(), expected) || !reflect.DeepEqual(input.Documents(), original) {
				t.Fatal("ragy-owned evidence aliases input/output")
			}
			if !reflect.DeepEqual(left.Documents(), original[:1]) ||
				!reflect.DeepEqual(right.Documents(), original[1:]) {
				t.Fatal("Merge mutated a source result set")
			}
			output[0].Meta.Tags[0] = "host mutation"
			if result.Documents()[0].Meta.Tags[0] != "host mutation" {
				t.Fatal("BYOT metadata was unexpectedly deep-cloned")
			}
		})
	}
}

type aggregationEventResolver struct{ events *[]string }

func (r aggregationEventResolver) Resolve(doc Document[aggregationMeta]) Identity {
	*r.events = append(*r.events, "resolve:"+doc.ID)
	return Identity{DocumentID: doc.ID, MergeKey: doc.Meta.Key}
}

type aggregationObservedSet struct {
	ResultSet[aggregationMeta]

	events *[]string
}

func (r aggregationObservedSet) Len() int {
	*r.events = append(*r.events, "right.Len")
	return r.ResultSet.Len()
}
func (r aggregationObservedSet) Documents() []Document[aggregationMeta] {
	*r.events = append(*r.events, "right.Documents")
	return r.ResultSet.Documents()
}

//nolint:gocognit // Keep the success and partial-error callback event sequences explicit.
func TestAggregationMergeCallbackOrder(t *testing.T) {
	for _, invalidLeft := range []bool{false, true} {
		t.Run(fmt.Sprintf("invalidLeft=%t", invalidLeft), func(t *testing.T) {
			// Arrange: a host ResultSet records callbacks, including the eager Len and
			// lazy Documents boundary. An invalid left must not access right documents.
			var events []string
			resolver := aggregationEventResolver{events: &events}
			leftDocs := []Document[aggregationMeta]{
				aggregationDocument(t, 0, 1, false),
				aggregationDocument(t, 1, 2, false),
			}
			rightDoc := aggregationDocument(t, 2, 3, false)
			if invalidLeft {
				leftDocs[1].ID = ""
			}
			left := NewResultSet(leftDocs, resolver)
			right := aggregationObservedSet{
				ResultSet: NewResultSet([]Document[aggregationMeta]{rightDoc}, nil),
				events:    &events,
			}
			// Act.
			result, err := left.Merge(right)
			// Assert.
			expected := []string{"right.Len", "resolve:doc-0", "resolve:doc-1", "right.Documents", "resolve:doc-2"}
			if invalidLeft {
				expected = []string{"right.Len", "resolve:doc-0"}
			}
			if !reflect.DeepEqual(events, expected) {
				t.Fatalf("callbacks: got %v want %v", events, expected)
			}
			if invalidLeft {
				if !errors.Is(err, ragy.ErrMissingID) {
					t.Fatalf("expected invalid left error, got %v", err)
				}
				if !reflect.DeepEqual(result.Documents(), leftDocs[:1]) {
					t.Fatalf("partial left result changed: %#v", result.Documents())
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}
