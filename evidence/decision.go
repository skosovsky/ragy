package evidence

// Finite wire limits apply to both capture and strict decode.
const (
	MaxRecordBytes = 4 << 20
	MaxTextRunes   = 64 << 10
	MaxRecordDepth = 32
	MaxItems       = 1024
)

// HostRevisions are explicit host assertions, never inferred provider identities.
type HostRevisions struct{ Model, Prompt, Config, Recipe string }
type DecisionQueryInput struct {
	Index                                     int
	Text                                      string
	Retrieved, Selected, Delivered, Uncertain bool
}
type DecisionContributor struct {
	QueryIndex int `json:"query_index"`
	Rank       int `json:"rank"`
}
type DecisionSelection struct {
	Index        int                   `json:"index"`
	Contributors []DecisionContributor `json:"contributors"`
	Delivered    bool                  `json:"delivered"`
	Uncertain    bool                  `json:"uncertain"`
}
type DecisionInput struct {
	Queries     []DecisionQueryInput
	Selected    []DecisionSelection
	Fusion      Status
	Sufficiency *bool
	Stop        string
	Revisions   HostRevisions
}
type DecisionQuery struct {
	Index     int  `json:"index"`
	Text      Text `json:"text"`
	Retrieved bool `json:"retrieved"`
	Selected  bool `json:"selected"`
	Delivered bool `json:"delivered"`
	Uncertain bool `json:"uncertain"`
}
type RevisionObservation struct {
	Model  Text `json:"model"`
	Prompt Text `json:"prompt"`
	Config Text `json:"config"`
	Recipe Text `json:"recipe"`
}
type Decision struct {
	State       CaptureState        `json:"state"`
	Queries     []DecisionQuery     `json:"queries"`
	Selected    []DecisionSelection `json:"selected"`
	Fusion      Status              `json:"fusion"`
	Sufficiency *bool               `json:"sufficiency"`
	Stop        string              `json:"stop"`
	Revisions   RevisionObservation `json:"revisions"`
}

func emptyDecision(state CaptureState) Decision {
	return Decision{
		State:       state,
		Queries:     nil,
		Selected:    nil,
		Fusion:      MissingObservation,
		Sufficiency: nil,
		Stop:        string(Unavailable),
		Revisions: RevisionObservation{
			Model:  hidden(Unavailable),
			Prompt: hidden(Unavailable),
			Config: hidden(Unavailable),
			Recipe: hidden(Unavailable),
		},
	}
}
func captureDecision(c *capture, input *DecisionInput) Decision {
	if input == nil {
		return emptyDecision(Unavailable)
	}
	if !c.policy.AllowDecisions {
		return emptyDecision(Omitted)
	}
	out := emptyDecision(Observed)
	out.Fusion = input.Fusion
	out.Stop = input.Stop
	if input.Sufficiency != nil {
		value := *input.Sufficiency
		out.Sufficiency = &value
	}
	for _, query := range input.Queries {
		text := hidden(Omitted)
		if c.policy.AllowQuery {
			text = textValue(query.Text)
		}
		out.Queries = append(
			out.Queries,
			DecisionQuery{
				Index:     query.Index,
				Text:      text,
				Retrieved: query.Retrieved,
				Selected:  query.Selected,
				Delivered: query.Delivered,
				Uncertain: query.Uncertain,
			},
		)
	}
	for _, selected := range input.Selected {
		item := selected
		item.Contributors = append([]DecisionContributor(nil), selected.Contributors...)
		out.Selected = append(out.Selected, item)
	}
	out.Revisions = RevisionObservation{
		Model:  c.identifier(ModelIdentifier, input.Revisions.Model),
		Prompt: c.identifier(PromptIdentifier, input.Revisions.Prompt),
		Config: c.identifier(ConfigIdentifier, input.Revisions.Config),
		Recipe: c.identifier(RecipeIdentifier, input.Revisions.Recipe),
	}
	return out
}
func validDecision(value Decision) bool {
	for _, text := range []Text{value.Revisions.Model, value.Revisions.Prompt, value.Revisions.Config, value.Revisions.Recipe} {
		if !validText(text) {
			return false
		}
	}
	if value.State != Observed {
		return (value.State == Omitted || value.State == Unavailable || value.State == Unsupported) &&
			len(value.Queries) == 0 &&
			len(value.Selected) == 0 &&
			value.Sufficiency == nil &&
			value.Fusion == MissingObservation &&
			value.Stop == string(Unavailable) &&
			value.Revisions.Model.State == Unavailable &&
			value.Revisions.Prompt.State == Unavailable &&
			value.Revisions.Config.State == Unavailable &&
			value.Revisions.Recipe.State == Unavailable
	}
	if len(value.Queries) > MaxItems || len(value.Selected) > MaxItems {
		return false
	}
	if value.Fusion != StageObserved && value.Fusion != NotRun && value.Fusion != MissingObservation {
		return false
	}
	switch value.Stop {
	case "assessed", "budget-exhausted", "deadline", "price-unavailable", "stage-failure":
	default:
		return false
	}
	for index, query := range value.Queries {
		if query.Index != index || !validText(query.Text) || query.Delivered && !query.Selected ||
			query.Selected && !query.Retrieved {
			return false
		}
	}
	return validDecisionSelections(value)
}
func validDecisionSelections(value Decision) bool {
	for index, selected := range value.Selected {
		if selected.Index != index || len(selected.Contributors) == 0 || len(selected.Contributors) > MaxItems ||
			value.Fusion != StageObserved {
			return false
		}
		seen := make(map[DecisionContributor]bool, len(selected.Contributors))
		for _, contributor := range selected.Contributors {
			if contributor.QueryIndex < 0 || contributor.QueryIndex >= len(value.Queries) || contributor.Rank < 1 ||
				seen[contributor] ||
				!value.Queries[contributor.QueryIndex].Selected {
				return false
			}
			seen[contributor] = true
			if selected.Delivered && !selected.Uncertain && !value.Queries[contributor.QueryIndex].Delivered {
				return false
			}
		}
	}
	return true
}
func boundedSnapshot(value Snapshot) bool {
	if len(value.Stages) > MaxItems || len(value.Diagnostics) > MaxItems {
		return false
	}
	for _, stage := range value.Stages {
		if len(stage.Hits) > MaxItems {
			return false
		}
		for _, hit := range stage.Hits {
			if len(hit.Sources) > MaxItems || len(hit.Locations) > MaxItems || len(hit.Contributions) > MaxItems {
				return false
			}
			for _, item := range hit.Contributions {
				if len(item.Locations) > MaxItems {
					return false
				}
			}
		}
	}
	return true
}

func boundedInput[TMeta any](input Input[TMeta]) bool {
	if len(input.Stages) > MaxItems || len(input.Diagnostics) > MaxItems {
		return false
	}
	if input.Decision != nil {
		if len(input.Decision.Queries) > MaxItems || len(input.Decision.Selected) > MaxItems {
			return false
		}
		for _, selected := range input.Decision.Selected {
			if len(selected.Contributors) > MaxItems {
				return false
			}
		}
	}
	for _, stage := range input.Stages {
		if len(stage.Hits) > MaxItems {
			return false
		}
		for _, hit := range stage.Hits {
			if len(hit.Sources) > MaxItems || len(hit.Locations) > MaxItems || len(hit.Contributions) > MaxItems {
				return false
			}
		}
	}
	return true
}
