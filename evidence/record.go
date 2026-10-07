package evidence

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// Record owns canonical bytes. Accessors return detached data. A decoded record
// validates shape, not the authenticity of the producer's export policy.
//

type Record struct{ data []byte }

func fromSnapshot(snapshot Snapshot) (Record, error) {
	if err := validateSnapshot(snapshot); err != nil {
		return Record{}, err
	}
	data, err := json.Marshal(snapshot)
	if err != nil {
		return Record{}, ragy.ErrInvalidArgument
	}
	if len(data) > MaxRecordBytes {
		return Record{}, ragy.ErrProtocol
	}
	return Record{data: data}, nil
}
func Decode(data []byte) (Record, error) {
	if len(data) > MaxRecordBytes || !utf8.Valid(data) {
		return Record{}, ragy.ErrProtocol
	}
	if err := uniqueFieldsDepth(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return Record{}, ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var snapshot Snapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return Record{}, ragy.ErrProtocol
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return Record{}, ragy.ErrProtocol
	}
	if snapshot.Schema != SchemaIdentity {
		return Record{}, ragy.ErrUnsupported
	}
	record, err := fromSnapshot(snapshot)
	if err != nil {
		return Record{}, err
	}
	var supplied, canonical any
	if json.Unmarshal(data, &supplied) != nil || json.Unmarshal(record.data, &canonical) != nil ||
		!sameShape(supplied, canonical) {
		return Record{}, ragy.ErrProtocol
	}
	return record, nil
}

func uniqueFieldsDepth(decoder *json.Decoder, depth int) error {
	if depth > MaxRecordDepth {
		return ragy.ErrProtocol
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, structured := token.(json.Delim)
	if !structured {
		return nil
	}
	switch delimiter {
	case '{':
		if objectErr := uniqueObjectFields(decoder, depth); objectErr != nil {
			return objectErr
		}
	case '[':
		for decoder.More() {
			if valueErr := uniqueFieldsDepth(decoder, depth+1); valueErr != nil {
				return valueErr
			}
		}
	default:
		return ragy.ErrProtocol
	}
	_, err = decoder.Token()
	return err
}
func (r Record) MarshalJSON() ([]byte, error) {
	if len(r.data) == 0 {
		return nil, ragy.ErrInvalidArgument
	}
	return slices.Clone(r.data), nil
}
func (r *Record) UnmarshalJSON(data []byte) error {
	if r == nil {
		return ragy.ErrInvalidArgument
	}
	decoded, err := Decode(data)
	if err != nil {
		return err
	}
	*r = decoded
	return nil
}
func (r Record) Snapshot() (Snapshot, error) {
	var snapshot Snapshot
	if len(r.data) == 0 || json.Unmarshal(r.data, &snapshot) != nil {
		return Snapshot{}, ragy.ErrInvalidArgument
	}
	return snapshot, nil
}

// sameShape rejects missing required fields even when their zero value is valid.
func sameShape(a, b any) bool {
	switch expected := b.(type) {
	case map[string]any:
		actual, ok := a.(map[string]any)
		if !ok || len(actual) != len(expected) {
			return false
		}
		for key, value := range expected {
			found, exists := actual[key]
			if !exists || !sameShape(found, value) {
				return false
			}
		}
	case []any:
		actual, ok := a.([]any)
		if !ok || len(actual) != len(expected) {
			return false
		}
		for i, value := range expected {
			if !sameShape(actual[i], value) {
				return false
			}
		}
	default:
	}
	return true
}

func validText(value Text) bool {
	switch value.State {
	case Observed:
		return value.Value != nil && *value.Value != "" && utf8.ValidString(*value.Value) &&
			utf8.RuneCountInString(*value.Value) <= MaxTextRunes
	case Omitted, Unavailable, Unsupported:
		return value.Value == nil
	default:
		return false
	}
}
func validNumber(value Number) bool {
	switch value.State {
	case Observed:
		return value.Value != nil && !math.IsNaN(*value.Value) && !math.IsInf(*value.Value, 0)
	case Omitted, Unavailable, Unsupported:
		return value.Value == nil
	default:
		return false
	}
}
func validSource(ref Source) bool {
	for _, field := range []Text{ref.Namespace, ref.ID, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
		if !validText(field) {
			return false
		}
	}
	return true
}
func capturedSource(ref Source) bool {
	for _, field := range []Text{ref.Namespace, ref.ID, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
		if !validText(field) || (field.State != Observed && field.State != Omitted) {
			return false
		}
	}
	return true
}
func validScore(score Score) bool {
	if !validText(score.Semantics) {
		return false
	}
	switch score.State {
	case ScoreNative, ScoreNormalized:
		if score.Semantics.State != Observed && score.Semantics.State != Omitted {
			return false
		}
		if score.Value == nil || math.IsNaN(*score.Value) || math.IsInf(*score.Value, 0) {
			return false
		}
		return score.State != ScoreNormalized || (*score.Value >= 0 && *score.Value <= 1)
	case ScoreAbsent:
		return score.Value == nil && score.Semantics.State == Unavailable
	case string(Omitted), string(Unavailable), string(Unsupported):
		return score.Value == nil
	default:
		return false
	}
}
func validLabel(label Label) bool {
	if !validText(label.Query) || !validText(label.Rubric) || !validNumber(label.Grade) || !validSource(label.Source) {
		return false
	}
	switch label.State {
	case string(Observed):
		return capturedSource(label.Source) && (label.Query.State == Observed || label.Query.State == Omitted) &&
			(label.Rubric.State == Observed || label.Rubric.State == Omitted) && (label.Grade.State == Observed || label.Grade.State == Omitted)
	case Ungradable:
		for _, field := range []Text{label.Source.Namespace, label.Source.ID, label.Source.Revision, label.Source.Transformation, label.Source.Artifact, label.Source.Representation} {
			if field.State != Unavailable {
				return false
			}
		}
		return label.Grade.State == Unavailable && label.Rubric.State == Unavailable && label.Query.State == Unavailable
	default:
		return false
	}
}
func validHit(hit WireHit) bool {
	if !validText(hit.ID) || hit.ID.State != Observed || !validNumber(hit.Rank) || !validScore(hit.Score) ||
		!validText(hit.Snippet) ||
		!validLabel(hit.Judgment) || !validLocations(hit) || !validContributions(hit) {
		return false
	}
	if hit.Rank.State == Observed && (*hit.Rank.Value <= 0 || *hit.Rank.Value > float64(MaxExactRank) ||
		math.Trunc(*hit.Rank.Value) != *hit.Rank.Value) {
		return false
	}
	if hit.SourcesState != Observed && hit.SourcesState != Unavailable && hit.SourcesState != Unsupported {
		return false
	}
	if hit.SourcesState != Observed && len(hit.Sources) != 0 {
		return false
	}
	if hit.SourcesState == Observed && len(hit.Sources) == 0 {
		return false
	}
	for _, ref := range hit.Sources {
		if !capturedSource(ref) {
			return false
		}
	}
	return true
}
func validStage(stage WireStage, index int, query Text) bool {
	if stage.Index != index || !validText(stage.Name) || stage.Name.State == Unavailable {
		return false
	}
	switch stage.Status {
	case StageObserved:
		if stage.HitsState != Observed && stage.HitsState != Omitted {
			return false
		}
	case NotRun, StageUnsupported, MissingObservation, StageUnavailable:
		if len(stage.Hits) != 0 || stage.HitsState != Unavailable {
			return false
		}
	default:
		return false
	}
	if stage.HitsState != Observed && len(stage.Hits) != 0 {
		return false
	}
	for _, hit := range stage.Hits {
		if !validHit(hit) || !labelBinding(query, hit) {
			return false
		}
	}
	return true
}
func validateSnapshot(snapshot Snapshot) error {
	if snapshot.Schema != SchemaIdentity {
		return ragy.ErrUnsupported
	}
	if !validDecision(snapshot.Decision) || !boundedSnapshot(snapshot) {
		return ragy.ErrProtocol
	}
	for _, field := range []Text{snapshot.RetrievalID, snapshot.Scope, snapshot.Publication, snapshot.Recipe, snapshot.Query} {
		if !validText(field) {
			return ragy.ErrProtocol
		}
	}
	switch snapshot.Outcome {
	case Complete, CompleteEmpty, Partial, Insufficient, Failed:
	default:
		return ragy.ErrProtocol
	}
	switch snapshot.Reason {
	case NoReason, Budget, Deadline, MissingEvidence, PartialTargets, TargetFailure:
	default:
		return ragy.ErrProtocol
	}
	if (snapshot.Outcome == Complete || snapshot.Outcome == CompleteEmpty) && snapshot.Reason != NoReason {
		return ragy.ErrProtocol
	}
	if snapshot.Outcome == Partial && snapshot.Reason == NoReason {
		return ragy.ErrProtocol
	}
	if _, err := snapshot.Coverage.MarshalJSON(); err != nil {
		return ragy.ErrProtocol
	}
	for _, diagnostic := range snapshot.Diagnostics {
		if !validDiagnostic(diagnostic) {
			return ragy.ErrProtocol
		}
	}
	for index, stage := range snapshot.Stages {
		if !validSnapshotStage(snapshot, stage, index) {
			return ragy.ErrProtocol
		}
	}
	return nil
}

func labelBinding(query Text, hit WireHit) bool {
	if hit.Judgment.State != string(Observed) {
		return true
	}
	if query.State == Observed && hit.Judgment.Query.State == Observed && *query.Value != *hit.Judgment.Query.Value {
		return false
	}
	for _, ref := range hit.Sources {
		if compatibleSource(ref, hit.Judgment.Source) {
			return true
		}
	}
	return false
}
func compatibleSource(a, b Source) bool {
	left := []Text{a.Namespace, a.ID, a.Revision, a.Transformation, a.Artifact, a.Representation}
	right := []Text{b.Namespace, b.ID, b.Revision, b.Transformation, b.Artifact, b.Representation}
	for i, field := range left {
		if field.State == Observed && right[i].State == Observed && *field.Value != *right[i].Value {
			return false
		}
	}
	return true
}

func validDiagnostic(diagnostic Diagnostic) bool {
	if !validNumber(diagnostic.Number) {
		return false
	}
	switch diagnostic.Kind {
	case ModelCalls, RetrievalCalls, InputTokens, OutputTokens, CostUnits, LatencyMillis:
	default:
		return false
	}
	if diagnostic.Number.State != Observed {
		return true
	}
	if *diagnostic.Number.Value < 0 {
		return false
	}
	return diagnostic.Kind == LatencyMillis || math.Trunc(*diagnostic.Number.Value) == *diagnostic.Number.Value
}

func uniqueObjectFields(decoder *json.Decoder, depth int) error {
	seen := make(map[string]bool)
	for decoder.More() {
		keyToken, keyErr := decoder.Token()
		if keyErr != nil {
			return keyErr
		}
		key, ok := keyToken.(string)
		if !ok || seen[key] {
			return ragy.ErrProtocol
		}
		seen[key] = true
		if valueErr := uniqueFieldsDepth(decoder, depth+1); valueErr != nil {
			return valueErr
		}
	}
	return nil
}

func validSnapshotStage(snapshot Snapshot, stage WireStage, index int) bool {
	return validStage(stage, index, snapshot.RetrievalID) && (snapshot.Outcome != CompleteEmpty || len(stage.Hits) == 0)
}
