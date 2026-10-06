package lifecycle

import (
	"context"
	"errors"

	ragy "github.com/skosovsky/ragy"
)

type ReuseReason string

const (
	ReuseConfirmed  ReuseReason = "confirmed"
	ReuseAbsent     ReuseReason = "absent"
	ReuseChanged    ReuseReason = "changed"
	ReuseIncomplete ReuseReason = "incomplete"
)

// ReuseDecision is an observation, not a lease or mutation authorization.
// Only Confirmed permits skipping an unchanged ingestion at the observation point.
type ReuseDecision struct {
	Reason      ReuseReason
	Publication string
}

func (d ReuseDecision) CanSkip() bool { return d.Reason == ReuseConfirmed }

// CheckReuse verifies exact identity/profile and backend inventory without writes.
// Any concurrent lifecycle generation change invalidates the decision; retry policy
// remains with the host. A changed ACL or missing volatile target cannot be skipped.
func (e *Executor[TPayload]) CheckReuse(
	ctx context.Context, desired Identity, targets []string,
) (ReuseDecision, error) {
	if err := desired.Validate(); err != nil {
		return ReuseDecision{}, err
	}
	requested, err := inventoryTargetSet(targets)
	if err != nil {
		return ReuseDecision{}, err
	}
	snapshot, err := e.load(ctx, desired.Namespace)
	if err != nil {
		return ReuseDecision{}, err
	}
	id := activePublication(snapshot, desired.Source)
	if id == "" {
		return ReuseDecision{Reason: ReuseAbsent, Publication: ""}, nil
	}
	manifest := snapshot.Manifests[manifestIndex(snapshot, id)]
	decision := ReuseDecision{Reason: ReuseChanged, Publication: id}
	if manifest.Tombstone || manifest.Identity != desired {
		return decision, nil
	}
	decision.Reason = ReuseIncomplete
	if manifest.Partial || !completeReuseProfile(manifest, requested) {
		return decision, nil
	}
	ready, err := e.inspectReuse(ctx, manifest)
	if err != nil {
		return ReuseDecision{}, err
	}
	if !ready {
		return decision, nil
	}
	current, err := e.load(ctx, desired.Namespace)
	if err != nil {
		return ReuseDecision{}, err
	}
	if current.Generation != snapshot.Generation {
		return ReuseDecision{}, ErrConflict
	}
	decision.Reason = ReuseConfirmed
	return decision, nil
}

func completeReuseProfile(manifest Manifest, requested map[string]struct{}) bool {
	if len(manifest.Targets) != len(requested) {
		return false
	}
	for _, target := range manifest.Targets {
		if _, exists := requested[target.Name]; !exists || target.State != TargetReady {
			return false
		}
	}
	return true
}

func (e *Executor[TPayload]) inspectReuse(ctx context.Context, manifest Manifest) (bool, error) {
	for _, target := range manifest.Targets {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		port := e.port(target.Name)
		if port == nil {
			return false, ragy.ErrUnsupported
		}
		result, err := port.Inspect(ctx, StageRequest{Manifest: cloneManifest(manifest), Target: target.Name})
		if err != nil {
			return false, errors.Join(ErrOutcomeUnknown, err)
		}
		if err = ctx.Err(); err != nil {
			return false, err
		}
		switch result.State {
		case TargetReady:
			if result.Revision != manifest.Identity.Revision {
				return false, ragy.ErrProtocol
			}
		case TargetPending, TargetFailed:
			if result.Revision != "" {
				return false, ragy.ErrProtocol
			}
			return false, nil
		case TargetUnknown:
			if result.Revision != "" {
				return false, ragy.ErrProtocol
			}
			return false, ErrOutcomeUnknown
		default:
			return false, ragy.ErrProtocol
		}
	}
	return true, nil
}
