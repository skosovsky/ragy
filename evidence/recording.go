package evidence

import (
	"context"
	"errors"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

var ErrRecordingFailed = errors.New("evidence recording failed")

type Mode string

const (
	Disabled   Mode = "disabled"
	BestEffort Mode = "best_effort"
	Required   Mode = "required"
)

type RecordingState string

const (
	RecordingDisabled RecordingState = "disabled"
	Recorded          RecordingState = "recorded"
	RecordingFailed   RecordingState = "failed"
)

type Receipt struct {
	State  RecordingState
	Record Record
}
type Execution[TResult any] struct {
	Result  TResult
	Receipt Receipt
}
type RecordingConfig[TResult any] struct {
	Mode        Mode
	Sink        Sink
	Execute     func(context.Context) (TResult, error)
	CloneResult func(TResult) (TResult, error)
	Capture     func(context.Context, access.Binding, TResult, error) (Record, error)
}

// RecordingError keeps a stable non-sensitive message; sink error text is never
// serialized into a receipt/record. [errors.Is]/[errors.As] can inspect the original cause.
type RecordingError struct{ cause error }

func (*RecordingError) Error() string     { return "evidence recording failed" }
func (e *RecordingError) Unwrap() []error { return []error{ErrRecordingFailed, e.cause} }

// Run executes retrieval exactly once. Required recording failure preserves the
// completed result/record and returns an error: overall success must not be claimed.
// Protection failure suppresses result and receipt, even after a sink write.
func Run[TResult any](
	ctx context.Context,
	read access.Binding,
	config RecordingConfig[TResult],
) (Execution[TResult], error) {
	if config.Execute == nil || !validMode(config.Mode) {
		return Execution[TResult]{}, ragy.ErrInvalidArgument
	}
	if config.Mode != Disabled && (nilSink(config.Sink) || config.CloneResult == nil || config.Capture == nil) {
		return Execution[TResult]{}, ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return Execution[TResult]{}, err
	}
	result, executionErr := config.Execute(ctx)
	if gateErr := read.Check(ctx); gateErr != nil {
		return Execution[TResult]{}, gateErr
	}
	if access.IsProtectionFailure(executionErr) {
		return Execution[TResult]{}, access.Protect(executionErr)
	}
	out := Execution[TResult]{Result: result, Receipt: Receipt{State: RecordingDisabled, Record: Record{data: nil}}}
	if config.Mode == Disabled {
		return out, executionErr
	}
	record, captureErr := captureResult(ctx, read, config, result, executionErr)
	if access.IsProtectionFailure(captureErr) {
		return Execution[TResult]{}, access.Protect(captureErr)
	}
	if gateErr := read.Check(ctx); gateErr != nil {
		return Execution[TResult]{}, gateErr
	}
	out.Receipt = Receipt{State: Recorded, Record: record}
	recordingErr := captureErr
	if captureErr == nil {
		recordingErr = config.Sink.Write(ctx, record)
	}
	if access.IsProtectionFailure(recordingErr) {
		return Execution[TResult]{}, access.Protect(recordingErr)
	}
	if gateErr := read.Check(ctx); gateErr != nil {
		return Execution[TResult]{}, gateErr
	}
	if recordingErr != nil {
		out.Receipt.State = RecordingFailed
		if config.Mode == Required {
			return out, errors.Join(executionErr, &RecordingError{cause: recordingErr})
		}
	}
	return out, executionErr
}

func captureResult[TResult any](
	ctx context.Context,
	read access.Binding,
	config RecordingConfig[TResult],
	result TResult,
	executionErr error,
) (Record, error) {
	if err := read.Check(ctx); err != nil {
		return Record{}, err
	}
	owned, err := config.CloneResult(result)
	if err != nil {
		return Record{}, err
	}
	if err = read.Check(ctx); err != nil {
		return Record{}, err
	}
	record, err := config.Capture(ctx, read, owned, executionErr)
	if err != nil {
		return Record{}, err
	}
	if _, err = record.Snapshot(); err != nil {
		return Record{}, err
	}
	return record, read.Check(ctx)
}
func validMode(mode Mode) bool {
	switch mode {
	case Disabled, BestEffort, Required:
		return true
	default:
		return false
	}
}
func nilSink(sink Sink) bool { return nilInterface(sink) }
func nilInterface(sink any) bool {
	if sink == nil {
		return true
	}
	value := reflect.ValueOf(sink)
	switch value.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
		return value.IsNil()
	case reflect.Invalid, reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128,
		reflect.Array, reflect.String, reflect.Struct, reflect.UnsafePointer:
		return false
	default:
		return false
	}
}
