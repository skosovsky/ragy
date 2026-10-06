package main

import (
	"encoding/json"
	"time"

	"example.com/ragyconsumer/internal/codexcall"
)

const (
	cliVersionDuration        = 3 * time.Second
	cliCalibrationInputMargin = 12000
	cliAttemptDuration        = 150 * time.Second
	cliCaptureDuration        = 30 * time.Minute
	cliCallDuration           = 45 * time.Second
	cliInputReservation       = 20000
	cliOutputReservation      = 2048
)

type codexHostProfile struct {
	CLIVersion                 string `json:"cli_version"`
	Model                      string `json:"model"`
	CalibrationSHA256          string `json:"calibration_sha256"`
	CallNanos                  int64  `json:"call_duration_nanos"`
	TokenPolicy                string `json:"token_policy"`
	PricePolicy                string `json:"price_policy"`
	HardTokenBoundVerified     bool   `json:"hard_token_bound_verified"`
	ProviderDispatchCountKnown bool   `json:"provider_dispatch_count_known"`
	ToolIsolation              string `json:"tool_isolation"`
}

func cliConfiguration() configuration {
	value := referenceConfiguration()
	value.CaptureDeadlineNanos = int64(cliCaptureDuration)
	value.DeadlineNanos = int64(cliAttemptDuration)
	value.ExtractionInputTokens, value.ExtractionOutputTokens = cliInputReservation, cliOutputReservation
	value.SummaryMapInput, value.SummaryReduceInput = cliInputReservation, cliInputReservation
	value.SummaryMapOutput, value.SummaryReduceOutput = cliOutputReservation, cliOutputReservation
	value.InputCap, value.OutputCap = cliInputReservation*globalCallCap, cliOutputReservation*globalCallCap
	value.CostCap, value.SummaryCallCost = 0, 0
	return value
}
func cliConfigurationBytes(profile *codexHostProfile) (json.RawMessage, string, error) {
	raw, err := json.Marshal(cliConfiguration())
	if err != nil {
		return nil, "", err
	}
	return raw, cliConfigurationIdentity(raw, profile), nil
}
func cliConfigurationIdentity(raw json.RawMessage, profile *codexHostProfile) string {
	encoded, err := json.Marshal(struct {
		Configuration json.RawMessage   `json:"configuration"`
		Profile       *codexHostProfile `json:"host_profile"`
	}{raw, profile})
	if err != nil {
		return ""
	}
	return digest(encoded)
}
func validateCaptureConfiguration(raw capture) error {
	if raw.HostProfile == nil {
		return validateConfiguration(raw.Configuration, raw.ConfigIdentity)
	}
	h := raw.HostProfile
	var value configuration
	if decodeStrict(raw.Configuration, &value) != nil || value != cliConfiguration() ||
		!validFingerprint(h.CalibrationSHA256) ||
		h.Model != raw.ModelIdentity ||
		h.CLIVersion == "" ||
		h.CallNanos != int64(cliCallDuration) ||
		h.TokenPolicy != "advisory-calibration-plus-input-byte-estimate" ||
		h.PricePolicy != "unknown-provider-price" ||
		h.HardTokenBoundVerified ||
		h.ProviderDispatchCountKnown ||
		h.ToolIsolation != "disabled-supported-tools-reject-observed-tools-additional-context-unverified" ||
		raw.ConfigIdentity != cliConfigurationIdentity(raw.Configuration, h) {
		return errInvalid
	}
	return nil
}
func reportedCLIUsage(receipts []codexcall.Result) (uint64, uint64, bool) {
	var input, output uint64
	known := true
	for _, call := range receipts {
		if call.Usage == nil {
			known = false
			continue
		}
		input += call.Usage.Input
		output += call.Usage.Output
	}
	return input, output, known
}

func graphConfigurationBytes(profile *codexHostProfile) (json.RawMessage, string, error) {
	if profile != nil {
		return cliConfigurationBytes(profile)
	}
	return configurationBytes()
}
func captureProfileDuration(profile *codexHostProfile) time.Duration {
	if profile != nil {
		return cliCaptureDuration
	}
	return graphCaptureDuration
}
func validLiveModelObservation(raw capture, sample observation) bool {
	if raw.ExecutionKind != liveExecution || (sample.Profile != communityProfile && sample.Profile != globalProfile) {
		return true
	}
	return validFingerprint(sample.Configuration) && (raw.HostProfile != nil || sample.TransportCallsKnown)
}

func profileBindingLifetime(profile *codexHostProfile) time.Duration {
	if profile != nil {
		return cliCaptureDuration
	}
	return 0
}
