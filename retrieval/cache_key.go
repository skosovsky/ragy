package retrieval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"sort"

	ragy "github.com/skosovsky/ragy"
)

// CacheIdentity declares host-owned index/recipe/configuration identities and
// requested capability profile. Rotate IndexRevision when a live index changes.
type CacheIdentity struct {
	Index         string   `json:"index"`
	IndexRevision string   `json:"index_revision"`
	Recipe        string   `json:"recipe"`
	Configuration string   `json:"configuration"`
	Capabilities  []string `json:"capabilities"`
}

func (i CacheIdentity) validate() error {
	if i.Index == "" || i.IndexRevision == "" || i.Recipe == "" || i.Configuration == "" || len(i.Capabilities) == 0 {
		return fmt.Errorf("%w: incomplete cache identity", ragy.ErrInvalidArgument)
	}
	if slices.Contains(i.Capabilities, "") {
		return fmt.Errorf("%w: empty cache capability", ragy.ErrInvalidArgument)
	}
	return nil
}

type graphCacheKey struct {
	Seeds      []string   `json:"seeds"`
	Direction  string     `json:"direction"`
	Depth      int        `json:"depth"`
	NodeFilter string     `json:"node_filter"`
	EdgeFilter string     `json:"edge_filter"`
	Page       *ragy.Page `json:"page"`
}

type requestCacheKey struct {
	Text            string            `json:"text"`
	EffectiveText   string            `json:"effective_text"`
	Binding         string            `json:"binding"`
	Host            []byte            `json:"host"`
	Identity        CacheIdentity     `json:"identity"`
	FetchLimit      int               `json:"fetch_limit"`
	TopK            int               `json:"top_k"`
	Threshold       *ScoreThreshold   `json:"threshold"`
	Filters         string            `json:"filters"`
	Vector          []float32         `json:"vector"`
	Graph           *graphCacheKey    `json:"graph"`
	Planned         bool              `json:"planned"`
	PlannedText     string            `json:"planned_text"`
	ExpandedText    string            `json:"expanded_text"`
	PlannedFilters  string            `json:"planned_filters"`
	PlannedRanges   []RangeConstraint `json:"planned_ranges"`
	PlannerCacheKey string            `json:"planner_cache_key"`
}

// RequestCacheKey hashes the complete core query/options/binding/profile identity.
// hostIdentity must project every BYOT intent/request-metadata value relevant to
// retrieval. Neither raw query nor raw policy is emitted as a cache-store key.
func RequestCacheKey[TIntent, TRequestMeta any](
	req Request[TIntent, TRequestMeta],
	identity CacheIdentity,
	hostIdentity []byte,
) (string, error) {
	if err := identity.validate(); err != nil {
		return "", err
	}
	if err := req.Options.Validate(); err != nil {
		return "", err
	}
	binding, err := req.Read.Fingerprint()
	if err != nil {
		return "", err
	}
	filters, err := req.Options.Filters.Fingerprint()
	if err != nil {
		return "", err
	}
	identity.Capabilities = append([]string(nil), identity.Capabilities...)
	sort.Strings(identity.Capabilities)
	key := requestCacheKey{
		Text:            req.Text,
		EffectiveText:   req.EffectiveText(),
		Binding:         binding,
		Host:            append([]byte(nil), hostIdentity...),
		Identity:        identity,
		FetchLimit:      req.Options.FetchLimit,
		TopK:            req.Options.TopK,
		Threshold:       req.Options.Threshold,
		Filters:         filters,
		Vector:          req.Options.Vector,
		Graph:           nil,
		Planned:         false,
		PlannedText:     "",
		ExpandedText:    "",
		PlannedFilters:  "",
		PlannedRanges:   nil,
		PlannerCacheKey: "",
	}
	if req.Plan != nil {
		key.Planned = true
		key.PlannedText = req.Plan.Text
		key.ExpandedText = req.Plan.ExpandedText
		key.PlannedRanges = req.Plan.Ranges
		key.PlannerCacheKey = req.Plan.CacheKey
		key.PlannedFilters, err = req.Plan.Filters.Fingerprint()
		if err != nil {
			return "", err
		}
	}
	if req.Options.Graph != nil {
		graphOptions := req.Options.Graph
		nodeFilter, nodeErr := graphOptions.NodeFilter.Fingerprint()
		if nodeErr != nil {
			return "", nodeErr
		}
		edgeFilter, edgeErr := graphOptions.EdgeFilter.Fingerprint()
		if edgeErr != nil {
			return "", edgeErr
		}
		key.Graph = &graphCacheKey{
			Seeds:      graphOptions.Seeds,
			Direction:  string(graphOptions.Direction),
			Depth:      graphOptions.Depth,
			NodeFilter: nodeFilter,
			EdgeFilter: edgeFilter,
			Page:       graphOptions.Page,
		}
	}
	data, err := json.Marshal(key)
	if err != nil {
		return "", fmt.Errorf("%w: invalid cache key payload", ragy.ErrInvalidArgument)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
