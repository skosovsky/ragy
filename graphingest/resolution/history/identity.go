package history

import (
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graphingest/resolution"
)

func requiredIdentities(values ...string) bool {
	for _, value := range values {
		if value == "" || !utf8.ValidString(value) {
			return false
		}
	}
	return true
}

func validDecision(decision resolution.Decision) bool {
	switch decision.State {
	case resolution.Resolved:
		return requiredIdentities(decision.Namespace, decision.Key, decision.Name)
	case resolution.Ambiguous:
		return decision.Namespace == "" && decision.Key == "" && decision.Name == ""
	default:
		return false
	}
}

func validateRecordIdentities[TKind, TRel comparable, TAttr any](record Record[TKind, TRel, TAttr]) error {
	if !requiredIdentities(record.Metadata.Run, record.Metadata.ExtractionFingerprint) ||
		(record.Metadata.Parent != "" && !validID(record.Metadata.Parent)) {
		return ragy.ErrInvalidArgument
	}
	for _, entity := range record.Input.Entities {
		if !requiredIdentities(entity.ID, entity.Name) || !utf8.ValidString(entity.Namespace) {
			return ragy.ErrInvalidArgument
		}
	}
	for _, relation := range record.Input.Relations {
		if !requiredIdentities(relation.ID, relation.From, relation.To) {
			return ragy.ErrInvalidArgument
		}
	}
	return validateResultIdentities(record.Result)
}

func validateResultIdentities[TKind, TRel comparable, TAttr any](result resolution.Result[TKind, TRel, TAttr]) error {
	if !requiredIdentities(result.OntologyIdentity, result.PolicyIdentity) {
		return ragy.ErrProtocol
	}
	for _, entity := range result.Entities {
		if !requiredIdentities(entity.ID) || !validDecision(entity.Identity) ||
			entity.Identity.State != resolution.Resolved {
			return ragy.ErrProtocol
		}
	}
	for _, relation := range result.Relations {
		if !requiredIdentities(relation.ID, relation.From, relation.To) {
			return ragy.ErrProtocol
		}
	}
	for _, unresolved := range result.Unresolved {
		if !requiredIdentities(unresolved.Mention) || (unresolved.Kind != "entity" && unresolved.Kind != "relation") {
			return ragy.ErrProtocol
		}
	}
	for _, trace := range result.EntityDecisions {
		if !requiredIdentities(trace.Mention) || !validDecision(trace.Identity) ||
			!canonicalIdentity(trace.Identity.State, trace.CanonicalID) {
			return ragy.ErrProtocol
		}
	}
	for _, trace := range result.RelationDecisions {
		if !validRelationTrace(trace) {
			return ragy.ErrProtocol
		}
	}
	return nil
}

func canonicalIdentity(state resolution.State, id string) bool {
	switch state {
	case resolution.Resolved:
		return requiredIdentities(id)
	case resolution.Ambiguous:
		return id == ""
	default:
		return false
	}
}

func validRelationTrace(trace resolution.RelationDecision) bool {
	if !requiredIdentities(trace.Mention) || !canonicalIdentity(trace.State, trace.CanonicalID) {
		return false
	}
	switch trace.State {
	case resolution.Resolved:
		return requiredIdentities(trace.From, trace.To, trace.Key)
	case resolution.Ambiguous:
		return utf8.ValidString(trace.From) && utf8.ValidString(trace.To) && trace.Key == ""
	default:
		return false
	}
}
