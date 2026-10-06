package source

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	ragy "github.com/skosovsky/ragy"
)

const mappingSchema = "ragy.source-mapping"

type mappingWire struct {
	Schema    string           `json:"schema"`
	State     string           `json:"state"`
	Text      string           `json:"text"`
	Fragments []MappedFragment `json:"fragments"`
}

// MarshalJSON encodes the owned snapshot with explicit unobserved state for the
// zero value. Display labels alone never become an exact mapping on the wire.
func (m MappedText) MarshalJSON() ([]byte, error) {
	wire := mappingWire{Schema: mappingSchema, State: "unobserved", Text: "", Fragments: nil}
	if m.text != "" || len(m.fragments) > 0 {
		if err := m.Validate(); err != nil {
			return nil, err
		}
		wire.State, wire.Text, wire.Fragments = "mapped", m.text, m.Fragments()
	}
	data, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("%w: source mapping encoding", ragy.ErrInvalidArgument)
	}
	return data, nil
}

// UnmarshalJSON validates structural mapping and snapshots support slices. It
// cannot certify the source text's authenticity: retained source resolution does
// that under the original read binding. Failed decoding preserves the old value.
func (m *MappedText) UnmarshalJSON(data []byte) error {
	if m == nil {
		return invalidLocation()
	}
	var wire mappingWire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return invalidLocation()
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return invalidLocation()
	}
	if wire.Schema != mappingSchema {
		return invalidLocation()
	}
	var snapshot MappedText
	switch wire.State {
	case "unobserved":
		if wire.Text != "" || len(wire.Fragments) != 0 {
			return invalidLocation()
		}
	case "mapped":
		snapshot = MappedText{text: wire.Text, fragments: wire.Fragments}
		if err := snapshot.Validate(); err != nil {
			return err
		}
	default:
		return invalidLocation()
	}
	*m = snapshot
	return nil
}
