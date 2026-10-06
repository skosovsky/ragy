//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"

	"github.com/skosovsky/ragy/adapters/openai/structured"
)

func newProviderExtractionPorts(
	ctx context.Context,
	cfg structured.Config,
	counter func(context.Context, []byte) (uint64, error),
) (extractionPorts, error) {
	if ctx == nil || counter == nil {
		return extractionPorts{}, errInvalid
	}
	if _, ok := ctx.Deadline(); !ok {
		return extractionPorts{}, errInvalid
	}
	cfg.CountTokens = func(request []byte) (uint64, error) { return counter(ctx, request) }
	cfg.Instructions = extractionInstructions
	cfg.SchemaName = "graph_extraction"
	cfg.Schema = json.RawMessage(extractionSchema)
	cfg.Validate = validateExtractionOutput
	cfg.Duration = attemptDuration
	cfg.MaxRequestBytes = summaryInputBytes
	cfg.MaxResponseBytes = summaryInputBytes
	client, transportCalls := observeModelClient(cfg.HTTPClient)
	cfg.HTTPClient = client
	model, err := structured.NewExtractor[string, string, graphAttributes](
		cfg,
		func(structured.Usage) (uint64, error) { return summaryCallCost, nil },
	)
	if err != nil {
		return extractionPorts{}, err
	}
	return extractionPorts{
		model:          model.Model,
		count:          model.CountInputTokens,
		transportCalls: transportCalls,
		configuration:  providerExtractionIdentity(cfg.Model, cfg.BaseURL),
		modelIdentity:  cfg.Model,
	}, nil
}

type extractionWireAttributes struct {
	Owner *string `json:"owner"`
}
type extractionWireEntity struct {
	ID         string                    `json:"id"`
	Name       string                    `json:"name"`
	Kind       string                    `json:"kind"`
	Attributes *extractionWireAttributes `json:"attributes"`
	Snippets   []int                     `json:"snippets"`
}
type extractionWireRelation struct {
	ID         string                    `json:"id"`
	From       string                    `json:"from"`
	To         string                    `json:"to"`
	Kind       string                    `json:"kind"`
	Attributes *extractionWireAttributes `json:"attributes"`
	Snippets   []int                     `json:"snippets"`
}

func validateExtractionOutput(raw json.RawMessage) error {
	var output struct {
		Entities  []extractionWireEntity   `json:"entities"`
		Relations []extractionWireRelation `json:"relations"`
	}
	if err := decodeStrict(raw, &output); err != nil || output.Entities == nil || output.Relations == nil {
		return errInvalid
	}
	if err := validateExtractionEntities(output.Entities); err != nil {
		return err
	}
	for _, edge := range output.Relations {
		if edge.ID == "" || edge.From == "" || edge.To == "" || edge.Kind == "" || edge.Snippets == nil ||
			edge.Attributes == nil ||
			edge.Attributes.Owner == nil {
			return errInvalid
		}
	}
	return nil
}
func validateExtractionEntities(entities []extractionWireEntity) error {
	for _, entity := range entities {
		if entity.ID == "" || entity.Name == "" || entity.Kind == "" || entity.Snippets == nil ||
			entity.Attributes == nil ||
			entity.Attributes.Owner == nil {
			return errInvalid
		}
	}
	return nil
}
