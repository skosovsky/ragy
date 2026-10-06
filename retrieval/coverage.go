package retrieval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"sort"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

// CoverageState distinguishes unobserved admission from complete/partial scope.
type CoverageState string

const (
	CoverageUnobserved   CoverageState = "unobserved"
	CoverageComplete     CoverageState = "complete"
	CoveragePartial      CoverageState = "partial"
	CoverageUnrestricted CoverageState = "unrestricted"
)

// ReadCoverage is an immutable admission report. Branch references are static
// configuration labels; no target document IDs/counts/content/error text is stored.
//

type ReadCoverage struct {
	state   CoverageState
	skipped []string
}

const CoverageSchemaIdentity = "ragy.read-coverage/admission"

type coverageWire struct {
	Schema  string        `json:"schema"`
	State   CoverageState `json:"state"`
	Skipped []string      `json:"skipped_branches"`
}

// MarshalJSON exports the immutable admission state without target payload data.
func (c ReadCoverage) MarshalJSON() ([]byte, error) {
	if err := c.validateWire(); err != nil {
		return nil, err
	}
	return json.Marshal(coverageWire{Schema: CoverageSchemaIdentity, State: c.State(), Skipped: c.SkippedBranches()})
}

// UnmarshalJSON owns its input and rejects incompatible schema/unknown fields.
func (c *ReadCoverage) UnmarshalJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var wire coverageWire
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("%w: coverage trailing input", ragy.ErrInvalidArgument)
	}
	if wire.Schema != CoverageSchemaIdentity {
		return fmt.Errorf("%w: coverage schema", ragy.ErrUnsupported)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if _, present := fields["skipped_branches"]; !present {
		return fmt.Errorf("%w: coverage required fields", ragy.ErrInvalidArgument)
	}
	if wire.State == "" {
		return fmt.Errorf("%w: coverage state", ragy.ErrInvalidArgument)
	}
	next := ReadCoverage{state: wire.State, skipped: append([]string(nil), wire.Skipped...)}
	if err := next.validateWire(); err != nil {
		return err
	}
	*c = next
	return nil
}

func (c ReadCoverage) validateWire() error {
	if c.State() == CoverageUnobserved && len(c.skipped) == 0 {
		return nil
	}
	return c.validate()
}

func (c ReadCoverage) State() CoverageState {
	if c.state == "" {
		return CoverageUnobserved
	}
	return c.state
}
func (c ReadCoverage) IsPartial() bool           { return c.State() == CoveragePartial }
func (c ReadCoverage) SkippedBranches() []string { return append([]string(nil), c.skipped...) }

// UnobservedReadCoverage reports that no admission observation was made.
func UnobservedReadCoverage() ReadCoverage {
	return ReadCoverage{state: CoverageUnobserved, skipped: nil}
}

// CompleteReadCoverage declares successful negotiation of all reachable leaves.
// It is a host adapter contract, not a certification of arbitrary custom Go code.
func CompleteReadCoverage() ReadCoverage { return ReadCoverage{state: CoverageComplete, skipped: nil} }

// PartialReadCoverage declares a pre-dispatch capability skip using a static label.
func PartialReadCoverage(branch string) (ReadCoverage, error) {
	if err := validateCoverageBranch(branch); err != nil {
		return UnobservedReadCoverage(), err
	}
	return ReadCoverage{state: CoveragePartial, skipped: []string{branch}}, nil
}

func validateCoverageBranch(branch string) error {
	if branch == "" || len(branch) > 128 {
		return fmt.Errorf("%w: static partial branch reference", ragy.ErrInvalidArgument)
	}
	for _, r := range branch {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' ||
			r == '.' {
			continue
		}
		return fmt.Errorf("%w: static partial branch reference", ragy.ErrInvalidArgument)
	}
	return nil
}

func (c ReadCoverage) validate() error {
	switch c.State() {
	case CoverageComplete, CoverageUnrestricted:
		if len(c.skipped) == 0 {
			return nil
		}
	case CoveragePartial:
		if len(c.skipped) == 0 {
			break
		}
		seen := map[string]struct{}{}
		for _, branch := range c.skipped {
			if err := validateCoverageBranch(branch); err != nil {
				return err
			}
			if _, duplicate := seen[branch]; duplicate {
				return fmt.Errorf("%w: duplicate coverage branch", ragy.ErrInvalidArgument)
			}
			seen[branch] = struct{}{}
		}
		return nil
	case CoverageUnobserved:
	}
	return fmt.Errorf("%w: read coverage", ragy.ErrInvalidArgument)
}

// MergeReadCoverage combines admitted leaf coverage, retaining every static skip label.
func MergeReadCoverage(a, b ReadCoverage) ReadCoverage {
	if a.State() == CoverageUnobserved {
		return ReadCoverage{state: b.state, skipped: b.SkippedBranches()}
	}
	if b.State() == CoverageUnobserved {
		return ReadCoverage{state: a.state, skipped: a.SkippedBranches()}
	}
	skipped := append(a.SkippedBranches(), b.skipped...)
	sort.Strings(skipped)
	skipped = slices.Compact(skipped)
	state := CoverageComplete
	if len(skipped) > 0 {
		state = CoveragePartial
	} else if a.State() == CoverageUnrestricted && b.State() == CoverageUnrestricted {
		state = CoverageUnrestricted
	}
	return ReadCoverage{state: state, skipped: skipped}
}

// BindPublicationCoverage retains immutable lifecycle exclusions alongside execution
// admission. Static target labels cannot be replaced by source/document diagnostics.
func BindPublicationCoverage(read access.Binding, coverage ReadCoverage) ReadCoverage {
	names := read.Publication().ExcludedTargets()
	if len(names) == 0 {
		return coverage
	}
	return MergeReadCoverage(coverage, ReadCoverage{state: CoveragePartial, skipped: names})
}
