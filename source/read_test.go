package source_test

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

type permission struct{ organization string }
type binaryPayload struct {
	mediaType string
	data      []byte
}
type binaryHost struct {
	reference      source.Reference
	permission     permission
	payload        binaryPayload
	loads          int
	wrongReference bool
	afterLoad      func()
}

func (h *binaryHost) Describe(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Descriptor[permission], error) {
	if len(request.References) == 0 || request.References[0] != h.reference {
		return nil, nil
	}
	return []source.Descriptor[permission]{{Reference: h.reference, Access: h.permission}}, nil
}
func (h *binaryHost) Load(_ context.Context, _ source.LookupRequest) ([]source.Materialized[binaryPayload], error) {
	h.loads++
	reference := h.reference
	if h.wrongReference {
		reference.Revision = "r2"
	}
	if h.afterLoad != nil {
		h.afterLoad()
	}
	return []source.Materialized[binaryPayload]{{Reference: reference, Payload: h.payload}}, nil
}

type binaryFixture struct {
	reader    *source.Reader[permission, binaryPayload]
	host      *binaryHost
	read      access.Binding
	revoked   *bool
	consumers *int
}

func newBinaryFixture(t *testing.T) binaryFixture {
	t.Helper()
	fields := filter.NewSchema()
	organization, err := fields.String("organization")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, organization, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	revoked, consumers := new(bool), new(int)
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "policy",
			PolicyEpoch: 7,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: access.CurrentPublication(),
		Now:         func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if *revoked {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	reference := retainedReference()
	reference.Representation = "original-image"
	host := &binaryHost{
		reference:  reference,
		permission: permission{organization: "a"},
		payload:    binaryPayload{mediaType: "image/png", data: []byte{0x89, 0x50, 0x4e, 0x47}},
	}
	reader, err := source.NewReader(source.ReadConfig[permission, binaryPayload]{
		Target: "images", Schema: schema, Catalog: host, Loader: host,
		Attributes: func(meta permission) (filter.RawAttributes, error) {
			return filter.RawAttributes{"organization": meta.organization}, nil
		},
		ValidatePayload: func(_ source.Reference, payload binaryPayload) error {
			*consumers++
			if payload.mediaType != "image/png" || len(payload.data) == 0 {
				return ragy.ErrProtocol
			}
			return nil
		},
		ClonePayload: func(payload binaryPayload) (binaryPayload, error) {
			*consumers++
			payload.data = bytes.Clone(payload.data)
			return payload, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return binaryFixture{reader: reader, host: host, read: read, revoked: revoked, consumers: consumers}
}

func TestSourceReaderOwnsTypedBinaryPayload(t *testing.T) {
	// Arrange.
	fixture := newBinaryFixture(t)
	request := source.LookupRequest{Read: fixture.read, References: []source.Reference{fixture.host.reference}}
	// Act.
	materialized, err := fixture.reader.Lookup(context.Background(), request)
	// Assert.
	if err != nil || len(materialized) != 1 || materialized[0].Reference != fixture.host.reference {
		t.Fatalf("materialization failed: %v", err)
	}
	materialized[0].Payload.data[0] = 0
	if fixture.host.payload.data[0] != 0x89 {
		t.Fatal("caller mutated retained original binary")
	}
	if fixture.host.loads != 1 {
		t.Fatal("materialization retried")
	}
}

func TestSourceReaderRejectsDeniedWrongRevisionAndRevokedBinaryBeforeConsumers(t *testing.T) {
	for _, kind := range []string{"denied", "wrong revision", "revoked during load"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			fixture := newBinaryFixture(t)
			switch kind {
			case "denied":
				fixture.host.permission.organization = "foreign"
			case "wrong revision":
				fixture.host.wrongReference = true
			case "revoked during load":
				fixture.host.afterLoad = func() { *fixture.revoked = true }
			}
			// Act.
			materialized, err := fixture.reader.Lookup(
				context.Background(),
				source.LookupRequest{Read: fixture.read, References: []source.Reference{fixture.host.reference}},
			)
			// Assert.
			if err == nil || len(materialized) != 0 || *fixture.consumers != 0 {
				t.Fatal("invalid binary reached a payload consumer")
			}
			if kind == "denied" && (!errors.Is(err, ragy.ErrUnavailable) || fixture.host.loads != 0) {
				t.Fatal("denied binary loaded")
			}
		})
	}
}
