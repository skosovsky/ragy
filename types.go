package ragy

import (
	"fmt"
)

// Page is an explicit pagination contract.
type Page struct {
	Limit  int `json:"limit"`
	Offset int `json:"offset"`
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
