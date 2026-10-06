package lifecycle

import (
	"context"
	"errors"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/observation"
)

type CleanupState string

const (
	CleanupWaiting CleanupState = "pending"
	CleanupUnknown CleanupState = "unknown"
	CleanupDone    CleanupState = "complete"
)

var ErrCleanupOverdue = errors.New("lifecycle cleanup overdue")
var ErrCleanupNotDue = errors.New("lifecycle cleanup not due")

// RetiredTarget identifies one exact inventory, not a reusable logical chunk ID.
type RetiredTarget struct {
	Manifest string       `json:"manifest"`
	Target   string       `json:"target"`
	State    CleanupState `json:"state"`
	Attempts uint64       `json:"attempts"`
	NextAt   time.Time    `json:"next_at"`
}

// CleanupJob is persisted independently from source publication visibility.
type CleanupJob struct {
	Owner     string          `json:"owner"`
	StartedAt time.Time       `json:"started_at"`
	Deadline  time.Time       `json:"deadline"`
	Overdue   bool            `json:"overdue"`
	Complete  bool            `json:"complete"`
	Items     []RetiredTarget `json:"items"`
}

// CleanupRequest supplies the exact managed inventory and current publication.
// Ports must fence concurrent stale writes, release only this inventory's supports,
// preserve other/host-owned support and honor their retained-read snapshot policy.
type CleanupRequest struct {
	Owner             Manifest
	Retired           Manifest
	Target            string
	ActivePublication string
	Retained          []Artifact
}

type CleanupPort interface {
	Cleanup(context.Context, CleanupRequest) (CleanupState, error)
	InspectCleanup(context.Context, CleanupRequest) (CleanupState, error)
}

type CleanupRegistration struct {
	Name string
	Port CleanupPort
}
type CleanupPolicy struct {
	Deadline time.Duration
	Backoff  []time.Duration
}
type CleanerConfig struct {
	Store   Store
	Targets []CleanupRegistration
	Now     func() time.Time
	Policy  CleanupPolicy
}

// Cleaner dispatches at most one operation per call; it never sleeps or retries.
// Validation/copying scale with namespace history; see lifecycle/README.md.
type Cleaner struct{ config CleanerConfig }

func NewCleaner(config CleanerConfig) (*Cleaner, error) {
	if nilPort(config.Store) || config.Now == nil || config.Policy.Deadline <= 0 ||
		len(config.Policy.Backoff) == 0 {
		return nil, ragy.ErrInvalidArgument
	}
	config.Targets = slices.Clone(config.Targets)
	config.Policy.Backoff = slices.Clone(config.Policy.Backoff)
	for _, delay := range config.Policy.Backoff {
		if delay <= 0 {
			return nil, ragy.ErrInvalidArgument
		}
	}
	names := make(map[string]struct{}, len(config.Targets))
	for _, target := range config.Targets {
		if !identities(target.Name) || nilPort(target.Port) {
			return nil, ragy.ErrInvalidArgument
		}
		if _, exists := names[target.Name]; exists {
			return nil, ragy.ErrInvalidArgument
		}
		names[target.Name] = struct{}{}
	}
	return &Cleaner{config: config}, nil
}

// Begin captures only the owner's exact previously published ancestry.
func (c *Cleaner) Begin(ctx context.Context, namespace, ownerID string) (CleanupJob, error) {
	ctx, span := observation.Begin(ctx, observation.StageLifecycleCleanupBegin)
	result, err := c.begin(ctx, namespace, ownerID)
	span.End(cleanupCompletion(result, err))
	return result, err
}

func (c *Cleaner) begin(ctx context.Context, namespace, ownerID string) (CleanupJob, error) {
	snapshot, err := c.load(ctx, namespace)
	if err != nil {
		return CleanupJob{}, err
	}
	ownerIndex := manifestIndex(snapshot, ownerID)
	if ownerIndex < 0 {
		return CleanupJob{}, ragy.ErrUnavailable
	}
	owner := snapshot.Manifests[ownerIndex]
	if owner.Retired {
		return CleanupJob{}, ErrRetired
	}
	if index := jobIndex(snapshot, ownerID); index >= 0 {
		return cloneJob(snapshot.Cleanups[index]), nil
	}
	state, err := confirmedState(owner.State, owner.Checkpoint)
	if err != nil || !afterPublished(state) {
		return CleanupJob{}, ragy.ErrInvalidArgument
	}
	if activePublication(snapshot, owner.Identity.Source) != ownerID {
		return CleanupJob{}, ErrConflict
	}
	job := CleanupJob{
		Owner:     ownerID,
		StartedAt: owner.PublishedAt,
		Deadline:  owner.PublishedAt.Add(c.config.Policy.Deadline),
		Overdue:   false,
		Complete:  false,
		Items:     nil,
	}
	job.Items, err = c.retiredItems(snapshot, owner)
	if err != nil {
		return CleanupJob{}, err
	}
	job.Complete = len(job.Items) == 0
	owner.State, owner.Checkpoint = CleanupPending, ""
	if job.Complete {
		owner.State = Complete
	}
	snapshot.Manifests[ownerIndex] = owner
	snapshot.Cleanups = append(snapshot.Cleanups, job)
	if _, err = c.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return CleanupJob{}, mutationError(err)
	}
	return cloneJob(job), nil
}

// Attempt executes one due destructive call; recovery explicitly permits overdue work.
func (c *Cleaner) Attempt(
	ctx context.Context,
	namespace, owner, retired, target string,
	recovery bool,
) (CleanupJob, error) {
	ctx, span := observation.Begin(ctx, observation.StageLifecycle)
	result, err := c.attempt(ctx, namespace, owner, retired, target, recovery)
	span.End(cleanupCompletion(result, err))
	return result, err
}

func (c *Cleaner) attempt(
	ctx context.Context,
	namespace, owner, retired, target string,
	recovery bool,
) (CleanupJob, error) {
	snapshot, jobAt, itemAt, request, port, err := c.operation(
		ctx,
		namespace,
		owner,
		retired,
		target,
	)
	if err != nil {
		return CleanupJob{}, err
	}
	job := snapshot.Cleanups[jobAt]
	item := job.Items[itemAt]
	if item.State == CleanupDone {
		return cloneJob(job), nil
	}
	now := c.config.Now().UTC()
	if !now.Before(job.Deadline) {
		job.Overdue = true
		if !recovery {
			snapshot.Cleanups[jobAt] = job
			if _, err = c.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
				return CleanupJob{}, mutationError(err)
			}
			return cloneJob(job), ErrCleanupOverdue
		}
	}
	if item.State == CleanupUnknown {
		return cloneJob(job), ErrOutcomeUnknown
	}
	if now.Before(item.NextAt) {
		return cloneJob(job), ErrCleanupNotDue
	}
	if item.Attempts == ^uint64(0) {
		return CleanupJob{}, ragy.ErrInvalidArgument
	}
	job.Items[itemAt].State = CleanupUnknown
	job.Items[itemAt].Attempts++
	snapshot.Cleanups[jobAt] = job
	if _, err = c.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return CleanupJob{}, mutationError(err)
	}
	if err = ctx.Err(); err != nil {
		return cloneJob(job), errors.Join(ErrOutcomeUnknown, err)
	}
	dispatchCtx, dispatchSpan := observation.Begin(ctx, observation.StageLifecycleCleanup)
	result, dispatchErr := port.Cleanup(dispatchCtx, request)
	dispatchSpan.End(cleanupStateCompletion(result, dispatchErr))
	if dispatchErr != nil {
		return cloneJob(job), errors.Join(ErrOutcomeUnknown, dispatchErr)
	}
	return c.confirm(ctx, namespace, owner, retired, target, result)
}

// Reconcile inspects uncertainty once and never issues another destructive call.
func (c *Cleaner) Reconcile(
	ctx context.Context,
	namespace, owner, retired, target string,
) (CleanupJob, error) {
	ctx, span := observation.Begin(ctx, observation.StageLifecycleCleanupReconcile)
	result, err := c.reconcile(ctx, namespace, owner, retired, target)
	span.End(cleanupCompletion(result, err))
	return result, err
}

func (c *Cleaner) reconcile(
	ctx context.Context,
	namespace, owner, retired, target string,
) (CleanupJob, error) {
	snapshot, jobAt, itemAt, request, port, err := c.operation(
		ctx,
		namespace,
		owner,
		retired,
		target,
	)
	if err != nil {
		return CleanupJob{}, err
	}
	job := snapshot.Cleanups[jobAt]
	if job.Items[itemAt].State != CleanupUnknown {
		return cloneJob(job), nil
	}
	inspectCtx, inspectSpan := observation.Begin(ctx, observation.StageLifecycleCleanupInspect)
	result, inspectErr := port.InspectCleanup(inspectCtx, request)
	inspectSpan.End(cleanupStateCompletion(result, inspectErr))
	if inspectErr != nil {
		return cloneJob(job), errors.Join(ErrOutcomeUnknown, inspectErr)
	}
	return c.confirm(ctx, namespace, owner, retired, target, result)
}

func (c *Cleaner) confirm(
	ctx context.Context,
	namespace, owner, retired, target string,
	state CleanupState,
) (CleanupJob, error) {
	snapshot, jobAt, itemAt, _, _, err := c.operation(ctx, namespace, owner, retired, target)
	if err != nil {
		return CleanupJob{}, errors.Join(ErrOutcomeUnknown, err)
	}
	job := snapshot.Cleanups[jobAt]
	if job.Items[itemAt].State != CleanupUnknown {
		return CleanupJob{}, ErrConflict
	}
	switch state {
	case CleanupWaiting, CleanupUnknown, CleanupDone:
	default:
		return cloneJob(job), errors.Join(ErrOutcomeUnknown, ragy.ErrProtocol)
	}
	item := &job.Items[itemAt]
	item.State = state
	if state == CleanupWaiting {
		backoff := item.Attempts - 1
		if backoff >= uint64(len(c.config.Policy.Backoff)) {
			backoff = uint64(len(c.config.Policy.Backoff)) - 1
		}
		item.NextAt = c.config.Now().UTC().Add(c.config.Policy.Backoff[backoff])
	}
	job.Complete = true
	for _, entry := range job.Items {
		if entry.State != CleanupDone {
			job.Complete = false
		}
	}
	snapshot.Cleanups[jobAt] = job
	if job.Complete {
		ownerAt := manifestIndex(snapshot, owner)
		snapshot.Manifests[ownerAt].State, snapshot.Manifests[ownerAt].Checkpoint = Complete, ""
	}
	if _, err = c.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return cloneJob(job), mutationError(err)
	}
	if err = ctx.Err(); err != nil {
		return cloneJob(job), errors.Join(ErrOutcomeUnknown, err)
	}
	if state == CleanupUnknown {
		return cloneJob(job), ErrOutcomeUnknown
	}
	return cloneJob(job), nil
}

func (c *Cleaner) load(ctx context.Context, namespace string) (Snapshot, error) {
	if c == nil || !identities(namespace) {
		return Snapshot{}, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	snapshot, err := c.config.Store.Load(ctx, namespace)
	if err != nil {
		return Snapshot{}, err
	}
	if snapshot.Namespace != namespace || snapshot.Validate() != nil {
		return Snapshot{}, ragy.ErrProtocol
	}
	if err = ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}
func (c *Cleaner) port(name string) CleanupPort {
	for _, registered := range c.config.Targets {
		if registered.Name == name {
			return registered.Port
		}
	}
	return nil
}
func (c *Cleaner) operation(
	ctx context.Context, namespace, owner, retired, target string,
) (Snapshot, int, int, CleanupRequest, CleanupPort, error) {
	snapshot, err := c.load(ctx, namespace)
	if err != nil {
		return Snapshot{}, 0, 0, CleanupRequest{}, nil, err
	}
	jobAt := jobIndex(snapshot, owner)
	if jobAt < 0 {
		return Snapshot{}, 0, 0, CleanupRequest{}, nil, ragy.ErrUnavailable
	}
	for itemAt, item := range snapshot.Cleanups[jobAt].Items {
		if item.Manifest == retired && item.Target == target {
			port := c.port(target)
			if port == nil {
				return Snapshot{}, 0, 0, CleanupRequest{}, nil, ragy.ErrUnsupported
			}
			ownerManifest := snapshot.Manifests[manifestIndex(snapshot, owner)]
			retiredManifest := snapshot.Manifests[manifestIndex(snapshot, retired)]
			if ownerManifest.Retired || retiredManifest.Retired {
				return Snapshot{}, 0, 0, CleanupRequest{}, nil, ErrRetired
			}
			request := CleanupRequest{
				Owner: cloneManifest(
					ownerManifest,
				), Retired: cloneManifest(retiredManifest), Target: target,
				ActivePublication: activePublication(
					snapshot,
					ownerManifest.Identity.Source,
				), Retained: nil,
			}
			request.Retained = retainedArtifacts(snapshot, retired, target)
			return snapshot, jobAt, itemAt, request, port, nil
		}
	}
	return Snapshot{}, 0, 0, CleanupRequest{}, nil, ragy.ErrInvalidArgument
}
func jobIndex(snapshot Snapshot, owner string) int {
	for i, job := range snapshot.Cleanups {
		if job.Owner == owner {
			return i
		}
	}
	return -1
}
func cloneJob(job CleanupJob) CleanupJob { job.Items = slices.Clone(job.Items); return job }

func retainedArtifacts(snapshot Snapshot, retired, target string) []Artifact {
	var artifacts []Artifact
	for _, manifest := range snapshot.Manifests {
		if manifest.ID == retired || cleanedInventory(snapshot, manifest.ID, target) {
			continue
		}
		captured := cloneManifest(manifest)
		for _, other := range captured.Targets {
			if other.Name == target {
				artifacts = append(artifacts, other.Artifacts...)
			}
		}
	}
	return artifacts
}

func (c *Cleaner) retiredItems(snapshot Snapshot, owner Manifest) ([]RetiredTarget, error) {
	manifests := make(map[string]Manifest, len(snapshot.Manifests))
	for _, manifest := range snapshot.Manifests {
		manifests[manifest.ID] = manifest
	}
	ancestors, err := retirementAncestry(manifests, owner)
	if err != nil {
		return nil, err
	}
	var items []RetiredTarget
	for _, retired := range snapshot.Manifests {
		if retired.Retired || !retirable(retired, owner, ancestors) {
			continue
		}
		for _, target := range retired.Targets {
			if c.port(target.Name) == nil {
				return nil, ragy.ErrUnsupported
			}
			items = append(
				items,
				RetiredTarget{
					Manifest: retired.ID,
					Target:   target.Name,
					State:    CleanupWaiting,
					Attempts: 0,
					NextAt:   owner.PublishedAt,
				},
			)
		}
	}
	return items, nil
}

func cleanedInventory(snapshot Snapshot, manifest, target string) bool {
	for _, job := range snapshot.Cleanups {
		for _, item := range job.Items {
			if item.Manifest == manifest && item.Target == target && item.State == CleanupDone {
				return true
			}
		}
	}
	return false
}
