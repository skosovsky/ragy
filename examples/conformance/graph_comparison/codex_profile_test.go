package main

import (
	"encoding/json"
	"testing"
)

func TestCodexGraphConfigurationBindsCalibratedProfile(t *testing.T) {
	// Arrange: frozen advisory controls retain the unchanged graph call/depth caps.
	profile := codexHostProfile{
		CLIVersion:        "calibrated-cli",
		Model:             "fixture-model",
		CalibrationSHA256: digest([]byte("calibration")),
		CallNanos: int64(
			cliCallDuration,
		),
		TokenPolicy:   "advisory-calibration-plus-input-byte-estimate",
		PricePolicy:   "unknown-provider-price",
		ToolIsolation: "disabled-supported-tools-reject-observed-tools-additional-context-unverified",
	}
	encoded, identity, err := cliConfigurationBytes(&profile)
	if err != nil {
		t.Fatal(err)
	}
	raw := capture{
		Configuration:  encoded,
		ConfigIdentity: identity,
		ModelIdentity:  profile.Model,
		HostProfile:    &profile,
	}
	// Act: validate original controls, then a changed executor with the old identity.
	validErr := validateCaptureConfiguration(raw)
	profile.CLIVersion = "changed-after-run"
	changedErr := validateCaptureConfiguration(raw)
	// Assert.
	if validErr != nil || changedErr == nil {
		t.Fatal(validErr, changedErr)
	}
	var controls configuration
	if json.Unmarshal(encoded, &controls) != nil {
		t.Fatal("invalid configuration")
	}
	reference := referenceConfiguration()
	if controls.LocalDepth != reference.LocalDepth || controls.LocalCalls != reference.LocalCalls ||
		controls.CommunityCalls != reference.CommunityCalls || controls.GlobalCalls != reference.GlobalCalls ||
		controls.DeadlineNanos != int64(cliAttemptDuration) {
		t.Fatal(controls)
	}
}
