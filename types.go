package ragy

import (
	"fmt"
)

// ClampScore bounds a public relevance score to [0, 1].
func ClampScore(v float64) float64 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

// Page is an explicit pagination contract.
type Page struct {
	Limit  int
	Offset int
}

// NewPage validates and constructs a page.
func NewPage(limit, offset int) (*Page, error) {
	p := &Page{Limit: limit, Offset: offset}
	if err := p.Validate(); err != nil {
		return nil, err
	}

	return p, nil
}

// Validate checks page invariants.
func (p *Page) Validate() error {
	if p == nil {
		return nil
	}

	if p.Limit <= 0 {
		return fmt.Errorf("%w: limit must be > 0", ErrInvalidPage)
	}

	if p.Offset < 0 {
		return fmt.Errorf("%w: offset must be >= 0", ErrInvalidPage)
	}

	return nil
}
