package observation_test

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/skosovsky/ragy/observation"
)

func ExampleObserverFunc() {
	// Host owns the queue and an event-drop unit distinct from core pair drops.
	queue := make(chan observation.Event, 1)
	var eventDrops atomic.Uint64
	session, err := observation.New(
		observation.Config{
			MaxEvents: 2,
			Observer: observation.ObserverFunc(func(ctx context.Context, event observation.Event) error {
				if err := ctx.Err(); err != nil {
					return err
				}
				select {
				case queue <- event:
				default:
					eventDrops.Add(1)
				}
				return nil
			}),
		},
	)
	if err != nil {
		panic(err)
	}
	_, span := observation.Begin(observation.WithSession(context.Background(), session), observation.StageRetrieval)
	span.End(observation.Finish(nil, observation.Count{Known: true, Value: 0}))
	// All producers have finished; only the host closes/drains the queue.
	close(queue)
	exported := 0
	for range queue {
		exported++
	}
	health := session.Stats()
	fmt.Println(health.Events, health.Dropped, health.Failures, exported, eventDrops.Load())
	// Output: 2 0 0 1 1
}
