// Package resolution groups typed extracted mentions under explicit host identity
// and ontology policies. It does not infer aliases, permissions or a universal ontology.
package resolution

import (
	"context"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

// Entity IDs and names are nonempty valid UTF-8. Namespace may be absent,
// but a nonempty namespace must be valid UTF-8. Core performs no normalization.
type Entity[TKind comparable, TAttr any] struct {
	ID         string           `json:"id"`
	Namespace  string           `json:"namespace"`
	Name       string           `json:"name"`
	Kind       TKind            `json:"kind"`
	Attributes TAttr            `json:"attributes"`
	Supports   []source.Locator `json:"supports"`
}

// Relation ID and endpoints are nonempty valid UTF-8 mention identities.
type Relation[TRel comparable, TAttr any] struct {
	ID         string           `json:"id"`
	From       string           `json:"from"`
	To         string           `json:"to"`
	Kind       TRel             `json:"kind"`
	Attributes TAttr            `json:"attributes"`
	Supports   []source.Locator `json:"supports"`
}
type Extraction[TKind, TRel comparable, TAttr any] struct {
	Entities  []Entity[TKind, TAttr]  `json:"entities"`
	Relations []Relation[TRel, TAttr] `json:"relations"`
}
type State string

const (
	Resolved  State = "resolved"
	Ambiguous State = "ambiguous"
)

// Decision requires nonempty valid UTF-8 namespace/key/name when Resolved.
// Name is canonical identity metadata, not a local mention label. Equal namespace/key
// decisions must agree on Name; conflicting names fail with ErrProtocol.
// Ambiguous requires all three fields empty. Malformed host decisions are ErrProtocol.
type Decision struct {
	State     State  `json:"state"`
	Namespace string `json:"namespace"`
	Key       string `json:"key"`
	Name      string `json:"name"`
}

// Variant retains conflicting attributes independently; no winner is selected.
type Variant[TKind comparable, TAttr any] struct {
	Kind       TKind            `json:"kind"`
	Attributes TAttr            `json:"attributes"`
	Supports   []source.Locator `json:"supports"`
}
type EntityGroup[TKind comparable, TAttr any] struct {
	ID       string                  `json:"id"`
	Identity Decision                `json:"identity"`
	Variants []Variant[TKind, TAttr] `json:"variants"`
}
type RelationGroup[TRel comparable, TAttr any] struct {
	ID       string                 `json:"id"`
	From     string                 `json:"from"`
	To       string                 `json:"to"`
	Variants []Variant[TRel, TAttr] `json:"variants"`
}

// Unresolved retains an ambiguous mention. Kind is the fact-category literal "entity" or "relation", not an ontology kind.
type Unresolved struct {
	Mention  string           `json:"mention"`
	Kind     string           `json:"kind"`
	Supports []source.Locator `json:"supports"`
}

// EntityDecision retains each local mention's policy outcome before grouping.
// CanonicalID is empty for ambiguity; conflicting variants remain in Result.
type EntityDecision struct {
	Mention     string           `json:"mention"`
	Identity    Decision         `json:"identity"`
	CanonicalID string           `json:"canonical_id"`
	Supports    []source.Locator `json:"supports"`
}

// RelationDecision records endpoint resolution and the explicit relation key.
// For ambiguous endpoints no relation key is requested or canonical ID invented.
type RelationDecision struct {
	Mention     string           `json:"mention"`
	State       State            `json:"state"`
	From        string           `json:"from"`
	To          string           `json:"to"`
	Key         string           `json:"key"`
	CanonicalID string           `json:"canonical_id"`
	Supports    []source.Locator `json:"supports"`
}
type Result[TKind, TRel comparable, TAttr any] struct {
	OntologyIdentity  string                       `json:"ontology_identity"`
	PolicyIdentity    string                       `json:"policy_identity"`
	Entities          []EntityGroup[TKind, TAttr]  `json:"entities"`
	Relations         []RelationGroup[TRel, TAttr] `json:"relations"`
	Unresolved        []Unresolved                 `json:"unresolved"`
	EntityDecisions   []EntityDecision             `json:"entity_decisions"`
	RelationDecisions []RelationDecision           `json:"relation_decisions"`
}

// Config ontology/policy identities and returned relation keys are nonempty valid UTF-8.
// Direct malformed inputs/configuration are ErrInvalidArgument; malformed host keys
// are ErrProtocol. Both suppress all output. Valid IDs retain JSON tuple + SHA256 framing.
// Config ports are explicit and model-free. Identity must implement namespace and
// alias rules; absent namespace is never guessed by the resolver. Equivalent compares
// typed attributes; conflicting variants and all supports remain in the result.
// Admission must authorize every original locator against the pinned read/catalog.
// CloneAttributes must deeply own BYOT values and be safe for concurrent use.
type Config[TKind, TRel comparable, TAttr any] struct {
	OntologyIdentity string
	PolicyIdentity   string
	MaxEntities      int
	MaxRelations     int
	MaxSupports      int
	ValidateEntity   func(TKind, TAttr) error
	ValidateRelation func(TRel, TKind, TKind, TAttr) error
	Identity         func(Entity[TKind, TAttr]) (Decision, error)
	RelationKey      func(Relation[TRel, TAttr]) (string, error)
	CloneAttributes  func(TAttr) (TAttr, error)
	Equivalent       func(TAttr, TAttr) bool
	AdmitSupport     func(context.Context, access.Binding, source.Locator) error
}
