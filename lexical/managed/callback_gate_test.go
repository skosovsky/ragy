//go:build darwin || linux

package managed_test

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type boundaryCodec struct {
	retrieval.MetadataCodec[metadata]

	active  bool
	calls   int
	trigger func() error
}

func (c *boundaryCodec) Encode(meta metadata) (filter.RawAttributes, error) {
	attrs, err := c.MetadataCodec.Encode(meta)
	if c.active {
		c.calls++
		if c.trigger != nil {
			return attrs, c.trigger()
		}
	}
	return attrs, err
}

func TestManagedCodecProtectionStopsCacheMissAndHit(t *testing.T) {
	for _, warm := range []bool{false, true} {
		for _, revoke := range []bool{false, true} {
			for _, fail := range []bool{false, true} {
				t.Run(fmt.Sprintf("warm=%t/revoke=%t/fail=%t", warm, revoke, fail), func(t *testing.T) {
					runManagedBoundary(t, warm, revoke, fail)
				})
			}
		}
	}
}

func runManagedBoundary(t *testing.T, warm, revoke, fail bool) {
	t.Helper()
	// Arrange: two published sources yield two authorized candidates.
	codec := &boundaryCodec{}
	f := newFixtureWithCodec(t, codec)
	codec.MetadataCodec = retrieval.NewJSONCodec[metadata](f.schema)
	for _, source := range []string{"policy", "faq"} {
		plan, records := sourcePlan(source+"1", source, "r1", "")
		f.ingest(t, plan, records, true)
	}
	read := f.pin(t)
	if warm {
		result, err := f.adapter.Retrieve(t.Context(), query(read))
		if err != nil || result.Len() != 2 {
			t.Fatal(result, err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	codec.active = true
	codec.trigger = func() error {
		if revoke {
			f.epoch = 8
		} else {
			cancel()
		}
		if fail {
			return fmt.Errorf("%w: private ordinary codec failure", ragy.ErrProtocol)
		}
		return nil
	}
	f.copies = nil
	// Act.
	result, err := f.adapter.Retrieve(ctx, query(read))
	// Assert: one metadata callback even on a cache hit, no clone or partial delivery.
	if fail && !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("callback cause lost", err)
	}
	cause := context.Canceled
	if revoke {
		cause = ragy.ErrUnavailable
	}
	if result.Len() != 0 || !access.IsProtectionFailure(err) || !errors.Is(err, cause) || codec.calls != 1 ||
		len(f.copies) != 0 {
		t.Fatal(result.Documents(), err, codec.calls, f.copies)
	}
}

func TestManagedNormalRankingAcrossCacheMissAndHit(t *testing.T) {
	// Arrange: same pinned complete predicate and two published source records.
	codec := &boundaryCodec{}
	f := newFixtureWithCodec(t, codec)
	codec.MetadataCodec = retrieval.NewJSONCodec[metadata](f.schema)
	for _, source := range []string{"policy", "faq"} {
		plan, records := sourcePlan(source+"1", source, "r1", "")
		f.ingest(t, plan, records, true)
	}
	read := f.pin(t)
	// Act.
	cold, coldErr := f.adapter.Retrieve(t.Context(), query(read))
	warm, warmErr := f.adapter.Retrieve(t.Context(), query(read))
	// Assert: hit and miss preserve scores, deterministic IDs, admitted metadata.
	if coldErr != nil || warmErr != nil || cold.Len() != 2 || warm.Len() != 2 {
		t.Fatal(coldErr, warmErr)
	}
	for i, doc := range cold.Documents() {
		other := warm.Documents()[i]
		if doc.ID != other.ID || doc.Score != other.Score || doc.Rank != other.Rank || other.Meta.Tenant != "a" {
			t.Fatal(doc, other)
		}
	}
}

func TestManagedCloneFailureAndRevocationStopsMissAndHit(t *testing.T) {
	for _, warm := range []bool{false, true} {
		t.Run(strconv.FormatBool(warm), func(t *testing.T) {
			// Arrange: codec only counts; the first payload clone revokes and fails.
			codec := &boundaryCodec{}
			f := newFixtureWithCodec(t, codec)
			codec.MetadataCodec = retrieval.NewJSONCodec[metadata](f.schema)
			for _, source := range []string{"policy", "faq"} {
				plan, records := sourcePlan(source+"1", source, "r1", "")
				f.ingest(t, plan, records, true)
			}
			read := f.pin(t)
			if warm {
				if _, err := f.adapter.Retrieve(t.Context(), query(read)); err != nil {
					t.Fatal(err)
				}
			}
			codec.active = true
			f.copies, f.revokeOnCopy, f.copyErr = nil, true, fmt.Errorf(
				"%w: private ordinary clone failure",
				ragy.ErrProtocol,
			)
			// Act.
			result, err := f.adapter.Retrieve(t.Context(), query(read))
			// Assert: post-clone authority check runs even on clone error.
			expectedEncodes := 1
			if warm {
				expectedEncodes = 2
			}
			if result.Len() != 0 || !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) ||
				len(f.copies) != 1 ||
				codec.calls != expectedEncodes {
				t.Fatal(result.Documents(), err, f.copies, codec.calls)
			}
		})
	}
}
