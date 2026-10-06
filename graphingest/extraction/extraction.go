package extraction

import (
	"context"
	"errors"
	"slices"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type Adapter[TAccess any, TKind, TRel comparable, TAttr any] struct {
	config Config[TAccess, TKind, TRel, TAttr]
}

func New[TAccess any, TKind, TRel comparable, TAttr any](
	cfg Config[TAccess, TKind, TRel, TAttr],
) (*Adapter[TAccess, TKind, TRel, TAttr], error) {
	if cfg.OntologyIdentity == "" || cfg.Configuration == "" || !cfg.Schema.IsFinalized() || cfg.MaxSnippets <= 0 ||
		cfg.MaxInputBytes <= 0 ||
		cfg.MaxEntities <= 0 ||
		cfg.MaxRelations <= 0 ||
		cfg.MaxSupports <= 0 ||
		cfg.Duration <= 0 ||
		cfg.Now == nil ||
		cfg.CloneAccess == nil ||
		cfg.Attributes == nil ||
		cfg.AdmitSnippet == nil ||
		cfg.CloneAttributes == nil ||
		cfg.ValidateEntity == nil ||
		cfg.ValidateRelation == nil ||
		cfg.Quote == nil ||
		cfg.CountInputTokens == nil ||
		cfg.Model == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Adapter[TAccess, TKind, TRel, TAttr]{config: cfg}, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) Extract(
	ctx context.Context,
	read access.Binding,
	ledger *budget.Ledger,
	input []Snippet[TAccess],
) (Result[TKind, TRel, TAttr], error) {
	if a == nil || ledger == nil {
		return Result[TKind, TRel, TAttr]{}, ragy.ErrInvalidArgument
	}
	deadline := a.config.Now().Add(a.config.Duration)
	shared, cancelShared := ledger.Context(ctx)
	defer cancelShared()
	child, cancel := context.WithTimeout(shared, deadline.Sub(a.config.Now()))
	defer cancel()
	clockGate := func() error {
		return checkExtractionClocks(child, ledger, a.config.Now, deadline)
	}
	gate := func() error {
		if err := clockGate(); err != nil {
			return err
		}
		return errors.Join(read.Check(child), clockGate())
	}
	if err := gate(); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	owned, modelInput, err := a.admit(child, read, input, gate, clockGate)
	if err = errors.Join(err, clockGate()); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	output, usage, err := a.call(child, ledger, modelInput, gate)
	if err = errors.Join(err, clockGate()); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	result, err := a.project(output, owned, gate)
	if err = errors.Join(err, clockGate()); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	if err = gate(); err != nil {
		return Result[TKind, TRel, TAttr]{}, err
	}
	return Result[TKind, TRel, TAttr]{Extraction: result, Usage: usage}, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) admit(
	ctx context.Context,
	read access.Binding,
	input []Snippet[TAccess],
	gate func() error,
	clockGate func() error,
) ([]Snippet[TAccess], ModelInput, error) {
	var empty ModelInput
	if len(input) == 0 || len(input) > a.config.MaxSnippets {
		return nil, empty, ragy.ErrInvalidArgument
	}
	mandatory, err := read.Prepare(
		ctx,
		a.config.Schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: false},
	)
	if err = errors.Join(err, clockGate()); err != nil {
		return nil, empty, err
	}
	if err = a.validateInput(input); err != nil {
		return nil, empty, err
	}
	owned := make([]Snippet[TAccess], len(input))
	modelInput := ModelInput{
		OntologyIdentity: a.config.OntologyIdentity,
		Configuration:    a.config.Configuration,
		Snippets:         make([]ModelSnippet, len(input)),
		MaxInputTokens:   0,
		MaxOutputTokens:  0,
	}
	for index, snippet := range input {
		snippet, err = a.admitSnippet(ctx, read, mandatory, snippet, gate)
		if err != nil {
			return nil, empty, err
		}
		owned[index] = snippet
		modelInput.Snippets[index] = ModelSnippet{Index: index, Text: snippet.Mapping.Text()}
	}
	return owned, modelInput, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) call(
	ctx context.Context,
	ledger *budget.Ledger,
	input ModelInput,
	gate func() error,
) (ModelOutput[TKind, TRel, TAttr], Usage, error) {
	var empty ModelOutput[TKind, TRel, TAttr]
	if err := gate(); err != nil {
		return empty, Usage{}, err
	}
	quote, err := a.config.Quote(ctx)
	err = errors.Join(err, gate())
	if err != nil {
		return empty, Usage{}, err
	}
	if err = gate(); err != nil {
		return empty, Usage{}, err
	}
	if quote.Kind != budget.Model || quote.Usage.InputTokens == 0 || quote.Usage.OutputTokens == 0 {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	input.MaxInputTokens, input.MaxOutputTokens = quote.Usage.InputTokens, quote.Usage.OutputTokens
	counterInput := input
	counterInput.Snippets = slices.Clone(input.Snippets)
	tokens, err := a.config.CountInputTokens(counterInput)
	err = errors.Join(err, gate())
	if err != nil {
		return empty, Usage{}, err
	}
	if err = gate(); err != nil {
		return empty, Usage{}, err
	}
	if tokens > quote.Usage.InputTokens {
		return empty, Usage{}, budget.ErrExhausted
	}
	lease, err := ledger.Reserve(ctx, quote)
	if err != nil {
		return empty, Usage{}, err
	}
	if err = gate(); err != nil {
		_ = lease.Settle(budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0}, false)
		return empty, Usage{}, err
	}
	// Client owns its input slice independently of quote/token counter state.
	input.Snippets = slices.Clone(input.Snippets)
	output, usage, callErr := a.config.Model(ctx, input)
	settleErr := lease.Settle(usage.Value, usage.Known && quote.CostKnown)
	if usage.Known &&
		(usage.Value.InputTokens > quote.Usage.InputTokens || usage.Value.OutputTokens > quote.Usage.OutputTokens) {
		settleErr = errors.Join(settleErr, budget.ErrUsageExceeded)
	}
	if err = errors.Join(callErr, settleErr, gate()); err != nil {
		return empty, Usage{}, err
	}
	if !quote.CostKnown {
		usage = Usage{Known: false, Value: budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0}}
	}
	return output, usage, nil
}
func evidence[TAccess any](indices []int, snippets []Snippet[TAccess], limit int) (string, []source.Locator, error) {
	if len(indices) == 0 || len(indices) > len(snippets) {
		return "", nil, ragy.ErrProtocol
	}
	seen := make(map[int]bool)
	var supports []source.Locator
	namespace := ""
	mixed := false
	for index, ordinal := range indices {
		if ordinal < 0 || ordinal >= len(snippets) || seen[ordinal] {
			return "", nil, ragy.ErrProtocol
		}
		seen[ordinal] = true
		snippet := snippets[ordinal]
		if index == 0 {
			namespace = snippet.Namespace
		} else if namespace != snippet.Namespace {
			mixed = true
		}
		for _, location := range snippet.Mapping.Supports() {
			if !slices.Contains(supports, location) {
				if len(supports) >= limit {
					return "", nil, ragy.ErrProtocol
				}
				supports = append(supports, location)
			}
		}
	}
	if mixed {
		namespace = ""
	}
	return namespace, supports, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) project(
	input ModelOutput[TKind, TRel, TAttr],
	snippets []Snippet[TAccess],
	gate func() error,
) (resolution.Extraction[TKind, TRel, TAttr], error) {
	var empty resolution.Extraction[TKind, TRel, TAttr]
	if len(input.Entities) > a.config.MaxEntities || len(input.Relations) > a.config.MaxRelations {
		return empty, ragy.ErrProtocol
	}
	out := resolution.Extraction[TKind, TRel, TAttr]{Entities: nil, Relations: nil}
	kinds := make(map[string]TKind)
	remaining := a.config.MaxSupports

	for _, entity := range input.Entities {
		if _, exists := kinds[entity.ID]; exists {
			return empty, ragy.ErrProtocol
		}
		projected, err := a.entity(entity, snippets, remaining, gate)
		if err != nil {
			return empty, err
		}
		kinds[entity.ID] = entity.Kind
		remaining -= len(projected.Supports)
		out.Entities = append(out.Entities, projected)
	}
	ids := make(map[string]bool)
	for _, edge := range input.Relations {
		if ids[edge.ID] {
			return empty, ragy.ErrProtocol
		}
		ids[edge.ID] = true
		from, fromOK := kinds[edge.From]
		to, toOK := kinds[edge.To]
		if !fromOK || !toOK {
			return empty, ragy.ErrProtocol
		}
		projected, err := a.relation(edge, from, to, snippets, remaining, gate)
		if err != nil {
			return empty, err
		}
		remaining -= len(projected.Supports)
		out.Relations = append(out.Relations, projected)
	}

	return out, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) validateInput(input []Snippet[TAccess]) error {
	bytesLeft, supportsLeft := a.config.MaxInputBytes, a.config.MaxSupports
	for _, snippet := range input {
		if !utf8.ValidString(snippet.Namespace) || snippet.Mapping.Validate() != nil ||
			len(snippet.Mapping.Text()) > bytesLeft {
			return ragy.ErrInvalidArgument
		}
		bytesLeft -= len(snippet.Mapping.Text())
		supports := snippet.Mapping.Supports()
		if len(supports) > supportsLeft {
			return ragy.ErrInvalidArgument
		}
		supportsLeft -= len(supports)
	}
	return nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) admitSnippet(
	ctx context.Context,
	read access.Binding,
	mandatory filter.Condition,
	snippet Snippet[TAccess],
	gate func() error,
) (Snippet[TAccess], error) {
	var err error
	if err = gate(); err != nil {
		return Snippet[TAccess]{}, err
	}
	snippet.Access, err = a.config.CloneAccess(snippet.Access)
	err = errors.Join(err, gate())
	if err != nil {
		return Snippet[TAccess]{}, err
	}
	if err = gate(); err != nil {
		return Snippet[TAccess]{}, err
	}
	attrs, attrErr := a.config.Attributes(snippet.Access)
	attrErr = errors.Join(attrErr, gate())
	if attrErr != nil {
		return Snippet[TAccess]{}, attrErr
	}
	if err = gate(); err != nil {
		return Snippet[TAccess]{}, err
	}
	if err = matchScope(a.config.Schema, attrs, mandatory); err != nil {
		return Snippet[TAccess]{}, err
	}
	err = a.config.AdmitSnippet(ctx, read, snippet)
	if err != nil {
		err = access.NonSkippable(err)
	}
	if err = errors.Join(err, gate()); err != nil {
		return Snippet[TAccess]{}, err
	}
	if err = gate(); err != nil {
		return Snippet[TAccess]{}, err
	}
	return snippet, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) entity(
	entity Entity[TKind, TAttr],
	snippets []Snippet[TAccess],
	remaining int,
	gate func() error,
) (resolution.Entity[TKind, TAttr], error) {
	var empty resolution.Entity[TKind, TAttr]
	if entity.ID == "" || !utf8.ValidString(entity.ID) || entity.Name == "" || !utf8.ValidString(entity.Name) {
		return empty, ragy.ErrProtocol
	}
	ns, supports, err := evidence(entity.Snippets, snippets, remaining)
	if err != nil {
		return empty, err
	}
	attrs, err := snapshotAttributes(
		entity.Attributes,
		a.config.CloneAttributes,
		func(attrs TAttr) error { return a.config.ValidateEntity(entity.Kind, attrs) },
		gate,
	)
	if err != nil {
		return empty, err
	}
	return resolution.Entity[TKind, TAttr]{
		ID:         entity.ID,
		Namespace:  ns,
		Name:       entity.Name,
		Kind:       entity.Kind,
		Attributes: attrs,
		Supports:   supports,
	}, nil
}

func (a *Adapter[TAccess, TKind, TRel, TAttr]) relation(
	edge Relation[TRel, TAttr],
	from, to TKind,
	snippets []Snippet[TAccess],
	remaining int,
	gate func() error,
) (resolution.Relation[TRel, TAttr], error) {
	var empty resolution.Relation[TRel, TAttr]
	if edge.ID == "" || !utf8.ValidString(edge.ID) {
		return empty, ragy.ErrProtocol
	}
	_, supports, err := evidence(edge.Snippets, snippets, remaining)
	if err != nil {
		return empty, err
	}
	attrs, err := snapshotAttributes(
		edge.Attributes,
		a.config.CloneAttributes,
		func(attrs TAttr) error { return a.config.ValidateRelation(edge.Kind, from, to, attrs) },
		gate,
	)
	if err != nil {
		return empty, err
	}
	return resolution.Relation[TRel, TAttr]{
		ID:         edge.ID,
		From:       edge.From,
		To:         edge.To,
		Kind:       edge.Kind,
		Attributes: attrs,
		Supports:   supports,
	}, nil
}

func snapshotAttributes[TAttr any](
	input TAttr,
	clone func(TAttr) (TAttr, error),
	validate func(TAttr) error,
	gate func() error,
) (TAttr, error) {
	var zero TAttr
	if err := gate(); err != nil {
		return zero, err
	}
	owned, err := clone(input)
	err = errors.Join(err, gate())
	if err != nil {
		return zero, err
	}
	if err = gate(); err != nil {
		return zero, err
	}
	if err = errors.Join(validate(owned), gate()); err != nil {
		return zero, err
	}
	if err = gate(); err != nil {
		return zero, err
	}
	owned, err = clone(input)
	err = errors.Join(err, gate())
	if err != nil {
		return zero, err
	}
	if err = gate(); err != nil {
		return zero, err
	}
	return owned, nil
}

func matchScope(schema filter.Schema, attrs filter.RawAttributes, mandatory filter.Condition) error {
	var err error
	attrs, err = schema.NormalizeAttributes(attrs)
	if err != nil {
		return err
	}
	if !filter.IsEmpty(mandatory.IR()) {
		allowed, matchErr := filter.MatchCondition(
			mandatory,
			func(field string) (any, bool) { value, exists := attrs[field]; return value, exists },
		)
		if matchErr != nil {
			return matchErr
		}
		if !allowed {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	return nil
}

func checkExtractionClocks(ctx context.Context, ledger *budget.Ledger, now func() time.Time, deadline time.Time) error {
	sharedErr := ledger.Check(ctx)
	var localErr error
	if !now().Before(deadline) {
		localErr = context.DeadlineExceeded
	}
	var contextErr error
	if err := ctx.Err(); err != nil {
		contextErr = access.Protect(err)
	}
	return errors.Join(sharedErr, localErr, contextErr)
}
