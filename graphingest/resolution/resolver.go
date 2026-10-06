package resolution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"sort"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

type Resolver[TKind, TRel comparable, TAttr any] struct{ config Config[TKind, TRel, TAttr] }

func New[TKind, TRel comparable, TAttr any](config Config[TKind, TRel, TAttr]) (*Resolver[TKind, TRel, TAttr], error) {
	if !requiredIdentities(config.OntologyIdentity, config.PolicyIdentity) || config.MaxEntities <= 0 ||
		config.MaxRelations <= 0 ||
		config.MaxSupports <= 0 ||
		config.ValidateEntity == nil ||
		config.ValidateRelation == nil ||
		config.Identity == nil ||
		config.RelationKey == nil ||
		config.CloneAttributes == nil ||
		config.Equivalent == nil ||
		config.AdmitSupport == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Resolver[TKind, TRel, TAttr]{config: config}, nil
}

func (r *Resolver[TKind, TRel, TAttr]) Resolve(
	ctx context.Context,
	read access.Binding,
	input Extraction[TKind, TRel, TAttr],
) (Result[TKind, TRel, TAttr], error) {
	if r == nil {
		return Result[TKind, TRel, TAttr]{}, ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	if err := r.admit(ctx, read, input); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	result := Result[TKind, TRel, TAttr]{
		OntologyIdentity: r.config.OntologyIdentity,
		PolicyIdentity:   r.config.PolicyIdentity, Entities: nil, Relations: nil, Unresolved: nil,
		EntityDecisions: nil, RelationDecisions: nil,
	}
	entities, err := r.entities(ctx, read, input.Entities, &result)
	if err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	if err = r.relations(ctx, read, input.Relations, entities, &result); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	sort.Slice(result.Entities, func(i, j int) bool { return result.Entities[i].ID < result.Entities[j].ID })
	sort.Slice(result.Relations, func(i, j int) bool { return result.Relations[i].ID < result.Relations[j].ID })
	if err = read.Check(ctx); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	return result, nil
}

func (r *Resolver[TKind, TRel, TAttr]) admit(
	ctx context.Context,
	read access.Binding,
	input Extraction[TKind, TRel, TAttr],
) error {
	if len(input.Entities) > r.config.MaxEntities || len(input.Relations) > r.config.MaxRelations {
		return ragy.ErrInvalidArgument
	}
	ids := make(map[string]bool)
	var all []source.Locator
	for _, entity := range input.Entities {
		if !requiredIdentities(entity.ID, entity.Name) || !utf8.ValidString(entity.Namespace) || ids[entity.ID] {
			return ragy.ErrInvalidArgument
		}
		ids[entity.ID] = true
		if err := boundedSupports(&all, entity.Supports, r.config.MaxSupports); err != nil {
			return err
		}
	}
	relations := make(map[string]bool)
	for _, relation := range input.Relations {
		if !requiredIdentities(relation.ID, relation.From, relation.To) || relations[relation.ID] ||
			!ids[relation.From] ||
			!ids[relation.To] {
			return ragy.ErrInvalidArgument
		}
		relations[relation.ID] = true
		if err := boundedSupports(&all, relation.Supports, r.config.MaxSupports); err != nil {
			return err
		}
	}
	// Validate the whole structural batch before invoking any authorization callback.
	for _, location := range all {
		if err := location.Validate(); err != nil {
			return err
		}
	}
	return r.admitLocations(ctx, read, all)
}

func (r *Resolver[TKind, TRel, TAttr]) admitLocations(
	ctx context.Context,
	read access.Binding,
	all []source.Locator,
) error {
	for _, location := range all {
		if err := read.Check(ctx); err != nil {
			return err
		}
		if !published(read.Publication(), location.Reference) {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
		if err := r.config.AdmitSupport(ctx, read, location); err != nil {
			return access.NonSkippable(err)
		}
		if err := read.Check(ctx); err != nil {
			return err
		}
	}
	return nil
}
func boundedSupports(all *[]source.Locator, supports []source.Locator, limit int) error {
	if len(supports) == 0 || len(supports) > limit-len(*all) {
		return ragy.ErrInvalidArgument
	}
	*all = append(*all, supports...)
	return nil
}
func published(publication access.Publication, reference source.Reference) bool {
	if publication.IsCurrent() {
		return true
	}
	for _, target := range publication.Targets() {
		if target.Namespace == reference.Namespace && target.Source == reference.Source &&
			target.Revision == reference.Revision &&
			target.AccessFingerprint == reference.AccessFingerprint {
			return true
		}
	}
	return false
}
func identityID(kind string, parts ...string) string {
	data, _ := json.Marshal(parts)
	digest := sha256.Sum256(data)
	return kind + ":" + hex.EncodeToString(digest[:])
}
func (r *Resolver[TKind, TRel, TAttr]) clone(ctx context.Context, read access.Binding, input TAttr) (TAttr, error) {
	var zero TAttr
	if err := read.Check(ctx); err != nil {
		return zero, err
	}
	output, err := r.config.CloneAttributes(input)
	if err != nil {
		return zero, err
	}
	if err = read.Check(ctx); err != nil {
		return zero, err
	}
	return output, nil
}

type resolvedEntity[TKind comparable] struct {
	id   string
	kind TKind
}

func (r *Resolver[TKind, TRel, TAttr]) entities(
	ctx context.Context,
	read access.Binding,
	input []Entity[TKind, TAttr],
	result *Result[TKind, TRel, TAttr],
) (map[string]resolvedEntity[TKind], error) {
	resolved := make(map[string]resolvedEntity[TKind])
	groups := make(map[string]int)
	for _, mention := range input {
		attributes, err := r.clone(ctx, read, mention.Attributes)
		if err != nil {
			return nil, err
		}
		decision, err := r.entityDecision(ctx, read, mention)
		if err != nil {
			return nil, err
		}
		resolved[mention.ID] = resolvedEntity[TKind]{id: "", kind: mention.Kind}
		trace := EntityDecision{
			Mention:     mention.ID,
			Identity:    decision,
			CanonicalID: "",
			Supports:    slices.Clone(mention.Supports),
		}
		if decision.State == Resolved {
			trace.CanonicalID = identityID("entity", decision.Namespace, decision.Key)
		}
		result.EntityDecisions = append(result.EntityDecisions, trace)
		if decision.State == Ambiguous {
			result.Unresolved = append(
				result.Unresolved,
				Unresolved{Mention: mention.ID, Kind: "entity", Supports: slices.Clone(mention.Supports)},
			)
			continue
		}
		id := identityID("entity", decision.Namespace, decision.Key)
		resolved[mention.ID] = resolvedEntity[TKind]{id: id, kind: mention.Kind}
		index, exists := groups[id]
		if !exists {
			index = len(result.Entities)
			groups[id] = index
			result.Entities = append(
				result.Entities,
				EntityGroup[TKind, TAttr]{ID: id, Identity: decision, Variants: nil},
			)
		}
		if result.Entities[index].Identity.Name != decision.Name {
			return nil, ragy.ErrProtocol
		}
		variants, err := mergeVariant(
			ctx,
			read,
			r.config.CloneAttributes,
			r.config.Equivalent,
			result.Entities[index].Variants,
			Variant[TKind, TAttr]{Kind: mention.Kind, Attributes: attributes, Supports: slices.Clone(mention.Supports)},
		)
		if err != nil {
			return nil, err
		}
		result.Entities[index].Variants = variants
	}
	return resolved, nil
}
func validateDecision(decision Decision) error {
	switch decision.State {
	case Resolved:
		if !requiredIdentities(decision.Namespace, decision.Key, decision.Name) {
			return ragy.ErrProtocol
		}
	case Ambiguous:
		if decision.Namespace != "" || decision.Key != "" || decision.Name != "" {
			return ragy.ErrProtocol
		}
	default:
		return ragy.ErrProtocol
	}
	return nil
}

func (r *Resolver[TKind, TRel, TAttr]) relations(
	ctx context.Context,
	read access.Binding,
	input []Relation[TRel, TAttr],
	entities map[string]resolvedEntity[TKind],
	result *Result[TKind, TRel, TAttr],
) error {
	groups := make(map[string]int)
	for _, mention := range input {
		from, fromOK := entities[mention.From]
		to, toOK := entities[mention.To]
		if !fromOK || !toOK {
			return ragy.ErrProtocol
		}
		if err := r.validateRelation(ctx, read, mention, from.kind, to.kind); err != nil {
			return err
		}
		if from.id == "" || to.id == "" {
			result.RelationDecisions = append(result.RelationDecisions, RelationDecision{
				Mention: mention.ID, State: Ambiguous, From: from.id, To: to.id,
				Key: "", CanonicalID: "", Supports: slices.Clone(mention.Supports),
			})
			result.Unresolved = append(
				result.Unresolved,
				Unresolved{Mention: mention.ID, Kind: "relation", Supports: slices.Clone(mention.Supports)},
			)
			continue
		}
		attributes, err := r.clone(ctx, read, mention.Attributes)
		if err != nil {
			return err
		}
		key, err := r.relationPolicy(ctx, read, mention)
		if err != nil {
			return err
		}
		id := identityID("relation", from.id, to.id, key)
		result.RelationDecisions = append(result.RelationDecisions, RelationDecision{
			Mention: mention.ID, State: Resolved, From: from.id, To: to.id,
			Key: key, CanonicalID: id, Supports: slices.Clone(mention.Supports),
		})
		index, exists := groups[id]
		if !exists {
			index = len(result.Relations)
			groups[id] = index
			result.Relations = append(
				result.Relations,
				RelationGroup[TRel, TAttr]{ID: id, From: from.id, To: to.id, Variants: nil},
			)
		}
		variants, err := mergeVariant(
			ctx,
			read,
			r.config.CloneAttributes,
			r.config.Equivalent,
			result.Relations[index].Variants,
			Variant[TRel, TAttr]{Kind: mention.Kind, Attributes: attributes, Supports: slices.Clone(mention.Supports)},
		)
		if err != nil {
			return err
		}
		result.Relations[index].Variants = variants
	}
	return nil
}

func mergeVariant[TKind comparable, TAttr any](
	ctx context.Context,
	read access.Binding,
	clone func(TAttr) (TAttr, error),
	equal func(TAttr, TAttr) bool,
	variants []Variant[TKind, TAttr],
	incoming Variant[TKind, TAttr],
) ([]Variant[TKind, TAttr], error) {
	for i, variant := range variants {
		if variant.Kind != incoming.Kind {
			continue
		}
		same, err := equivalent(ctx, read, clone, equal, variant.Attributes, incoming.Attributes)
		if err != nil {
			return nil, err
		}
		if same {
			for _, support := range incoming.Supports {
				if !slices.Contains(variants[i].Supports, support) {
					variants[i].Supports = append(variants[i].Supports, support)
				}
			}
			return variants, nil
		}
	}
	return append(variants, incoming), nil
}

func (r *Resolver[TKind, TRel, TAttr]) entityDecision(
	ctx context.Context,
	read access.Binding,
	mention Entity[TKind, TAttr],
) (Decision, error) {
	policyInput := mention
	var err error
	policyInput.Attributes, err = r.clone(ctx, read, mention.Attributes)
	if err != nil {
		return Decision{}, err
	}
	policyInput.Supports = slices.Clone(mention.Supports)
	if err = r.config.ValidateEntity(policyInput.Kind, policyInput.Attributes); err != nil {
		return Decision{}, err
	}
	if err = read.Check(ctx); err != nil {
		return Decision{}, err
	}
	decision, err := r.config.Identity(policyInput)
	if err != nil {
		return Decision{}, err
	}
	if err = read.Check(ctx); err != nil {
		return Decision{}, err
	}
	if err = validateDecision(decision); err != nil {
		return Decision{}, err
	}
	return decision, nil
}

func (r *Resolver[TKind, TRel, TAttr]) relationPolicy(
	ctx context.Context,
	read access.Binding,
	mention Relation[TRel, TAttr],
) (string, error) {
	policyInput := mention
	var err error
	policyInput.Attributes, err = r.clone(ctx, read, mention.Attributes)
	if err != nil {
		return "", err
	}
	policyInput.Supports = slices.Clone(mention.Supports)

	key, err := r.config.RelationKey(policyInput)
	if err != nil {
		return "", err
	}
	if err = read.Check(ctx); err != nil {
		return "", err
	}
	if !requiredIdentities(key) {
		return "", ragy.ErrProtocol
	}
	return key, nil
}

func equivalent[TAttr any](
	ctx context.Context,
	read access.Binding,
	clone func(TAttr) (TAttr, error),
	equal func(TAttr, TAttr) bool,
	first, second TAttr,
) (bool, error) {
	if err := read.Check(ctx); err != nil {
		return false, err
	}
	left, err := clone(first)
	if err != nil {
		return false, err
	}
	if err = read.Check(ctx); err != nil {
		return false, err
	}
	right, err := clone(second)
	if err != nil {
		return false, err
	}
	if err = read.Check(ctx); err != nil {
		return false, err
	}
	same := equal(left, right)
	if err = read.Check(ctx); err != nil {
		return false, err
	}
	return same, nil
}

func (r *Resolver[TKind, TRel, TAttr]) validateRelation(
	ctx context.Context,
	read access.Binding,
	mention Relation[TRel, TAttr],
	fromKind, toKind TKind,
) error {
	attributes, err := r.clone(ctx, read, mention.Attributes)
	if err != nil {
		return err
	}
	if err = r.config.ValidateRelation(mention.Kind, fromKind, toKind, attributes); err != nil {
		return err
	}
	return read.Check(ctx)
}

func requiredIdentities(values ...string) bool {
	for _, value := range values {
		if value == "" || !utf8.ValidString(value) {
			return false
		}
	}
	return true
}
