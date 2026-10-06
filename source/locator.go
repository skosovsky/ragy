package source

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// LocationKind identifies the representation addressed by a locator.
type LocationKind string

const (
	DocumentLocation LocationKind = "document"
	TextLocation     LocationKind = "text"
	PageLocation     LocationKind = "page"
	RegionLocation   LocationKind = "region"
	CellLocation     LocationKind = "table_cell"
	ImageLocation    LocationKind = "image_region"
)

// ByteSpan is a zero-based, half-open UTF-8 byte interval in Reference.Representation.
type ByteSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// ValidateText verifies bounds and code point boundaries against retained text.
func (s ByteSpan) ValidateText(text string) error {
	if !utf8.ValidString(text) || s.Start < 0 || s.End <= s.Start || s.End > len(text) {
		return invalidLocation()
	}
	if !utf8.RuneStart(text[s.Start]) || (s.End < len(text) && !utf8.RuneStart(text[s.End])) {
		return invalidLocation()
	}
	return nil
}

// PageGeometry separates physical page order from its printed label. Coordinates
// are points from the top-left of the unrotated page. Rotation is clockwise.
type PageGeometry struct {
	PhysicalIndex int     `json:"physical_index"`
	PrintedLabel  string  `json:"printed_label"`
	Width         float64 `json:"width_pt"`
	Height        float64 `json:"height_pt"`
	Rotation      int     `json:"rotation_deg"`
}

const (
	quarterTurn      = 90
	halfTurn         = 180
	threeQuarterTurn = 270
)

// Validate rejects unsupported or invalid geometry; it never guesses units.
func (p PageGeometry) Validate() error {
	if p.PhysicalIndex < 0 || !utf8.ValidString(p.PrintedLabel) || !positiveFinite(p.Width) ||
		!positiveFinite(p.Height) {
		return invalidLocation()
	}
	switch p.Rotation {
	case 0, quarterTurn, halfTurn, threeQuarterTurn:
		return nil
	default:
		return invalidLocation()
	}
}

// Point is expressed in page points.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Display converts an unrotated point into rotated display coordinates.
func (p PageGeometry) Display(point Point) (Point, error) {
	if err := p.Validate(); err != nil {
		return Point{}, err
	}
	if !inside(point, p.Width, p.Height) {
		return Point{}, invalidLocation()
	}
	switch p.Rotation {
	case quarterTurn:
		return Point{X: p.Height - point.Y, Y: point.X}, nil
	case halfTurn:
		return Point{X: p.Width - point.X, Y: p.Height - point.Y}, nil
	case threeQuarterTurn:
		return Point{X: point.Y, Y: p.Width - point.X}, nil
	default:
		return point, nil
	}
}

// Original is the inverse display-coordinate transformation.
func (p PageGeometry) Original(point Point) (Point, error) {
	if err := p.Validate(); err != nil {
		return Point{}, err
	}
	width, height := p.Width, p.Height
	if p.Rotation == quarterTurn || p.Rotation == threeQuarterTurn {
		width, height = height, width
	}
	if !inside(point, width, height) {
		return Point{}, invalidLocation()
	}
	switch p.Rotation {
	case quarterTurn:
		return Point{X: point.Y, Y: p.Height - point.X}, nil
	case halfTurn:
		return Point{X: p.Width - point.X, Y: p.Height - point.Y}, nil
	case threeQuarterTurn:
		return Point{X: p.Width - point.Y, Y: point.X}, nil
	default:
		return point, nil
	}
}

// Rectangle is a top-left, unrotated point rectangle.
type Rectangle struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Right  float64 `json:"right"`
	Bottom float64 `json:"bottom"`
}

// Validate checks positive area entirely within the given page.
func (r Rectangle) Validate(page PageGeometry) error {
	if err := page.Validate(); err != nil {
		return err
	}
	if !inside(Point{X: r.Left, Y: r.Top}, page.Width, page.Height) ||
		!inside(Point{X: r.Right, Y: r.Bottom}, page.Width, page.Height) || r.Right <= r.Left || r.Bottom <= r.Top {
		return invalidLocation()
	}
	return nil
}

// TableCell addresses one logical cell; spans do not create extra citations.
type TableCell struct {
	Table      string `json:"table"`
	Element    string `json:"element"`
	Row        int    `json:"row"`
	Column     int    `json:"column"`
	RowSpan    int    `json:"row_span"`
	ColumnSpan int    `json:"column_span"`
}

func (c TableCell) validate() error {
	if c.Table == "" || c.Element == "" || !utf8.ValidString(c.Table) || !utf8.ValidString(c.Element) ||
		c.Row < 0 || c.Column < 0 || c.RowSpan <= 0 || c.ColumnSpan <= 0 {
		return invalidLocation()
	}
	return nil
}

// Locator is a tagged value-only location in one retained source representation.
// Inactive fields must be zero. There is no rank, current-revision alias or parser
// pointer in identity. Text bounds are checked by ValidateText at resolution.
type Locator struct {
	Reference Reference    `json:"reference"`
	Kind      LocationKind `json:"kind"`
	Span      ByteSpan     `json:"span"`
	Page      PageGeometry `json:"page"`
	Region    Rectangle    `json:"region"`
	Cell      TableCell    `json:"cell"`
}

// Validate checks the tagged union and representation identity without source I/O.
func (l Locator) Validate() error {
	if err := l.Reference.Validate(); err != nil {
		return err
	}
	if err := l.validateActive(); err != nil {
		return err
	}
	var zero Locator
	inactive := l
	inactive.Reference = zero.Reference
	inactive.Kind = ""
	switch l.Kind {
	case DocumentLocation:
	case TextLocation:
		inactive.Span = zero.Span
	case PageLocation:
		inactive.Page = zero.Page
	case RegionLocation, ImageLocation:
		inactive.Page = zero.Page
		inactive.Region = zero.Region
	case CellLocation:
		inactive.Page = zero.Page
		inactive.Cell = zero.Cell
	default:
		return invalidLocation()
	}
	if inactive != zero {
		return invalidLocation()
	}
	return nil
}

func (l Locator) validateActive() error {
	switch l.Kind {
	case DocumentLocation:
		return nil
	case TextLocation:
		if l.Span.Start < 0 || l.Span.End <= l.Span.Start {
			return invalidLocation()
		}
		return nil
	case PageLocation:
		return l.Page.Validate()
	case RegionLocation, ImageLocation:
		return l.Region.Validate(l.Page)
	case CellLocation:
		if err := l.Page.Validate(); err != nil {
			return err
		}
		return l.Cell.validate()
	default:
		return invalidLocation()
	}
}

// Identity is the canonical revision/location identity. Ordering/reranking does
// not enter this value; equal logical merged cells yield the same identity.
func (l Locator) Identity() (string, error) {
	if err := l.Validate(); err != nil {
		return "", err
	}
	canonical := l
	canonical.Region = Rectangle{
		Left:   canonicalCoordinate(l.Region.Left),
		Top:    canonicalCoordinate(l.Region.Top),
		Right:  canonicalCoordinate(l.Region.Right),
		Bottom: canonicalCoordinate(l.Region.Bottom),
	}
	data, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("%w: locator encoding", ragy.ErrInvalidArgument)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

// ExtendedLocator carries a typed, host-owned extension alongside a canonical
// location. The host supplies ownership/validation of TExtension; the extension
// cannot change the retained reference or canonical citation identity.
type ExtendedLocator[TExtension any] struct {
	Location  Locator    `json:"location"`
	Extension TExtension `json:"extension"`
}

func positiveFinite(value float64) bool { return value > 0 && finite(value) }
func finite(value float64) bool         { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func inside(point Point, width, height float64) bool {
	return finite(point.X) && finite(point.Y) && point.X >= 0 && point.Y >= 0 && point.X <= width && point.Y <= height
}
func invalidLocation() error { return fmt.Errorf("%w: source locator", ragy.ErrInvalidArgument) }

func canonicalCoordinate(value float64) float64 {
	if value == 0 {
		return 0
	}
	return value
}
