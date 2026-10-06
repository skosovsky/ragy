package main

import (
	"errors"
	"testing"

	"github.com/skosovsky/ragy/recipe"
)

func TestCaptureConfigurationRejectsUnmatchedExecutionProfile(t *testing.T) {
	for _, scenario := range []string{"missing", "seed", "corpus-identity", "tokens", "calls", "fingerprint"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: complete valid observation grid, changing configuration only.
			raw := comparisonCapture(t)
			switch scenario {
			case "missing":
				raw.Configuration = experimentConfiguration{}
			case "seed":
				raw.Configuration.SeedPolicy = "fixed-seed-claimed"
			case "corpus-identity":
				raw.Configuration.CorpusSHA256 = "other-corpus"
			case "tokens":
				raw.Configuration.InputCap++
			case "calls":
				raw.Configuration.ModelLimits[1]++
			case "fingerprint":
				raw.ConfigIdentity = "unverified-label"
			}
			if scenario != "fingerprint" {
				raw.ConfigIdentity = configurationIdentity(raw.Configuration)
			}
			// Act.
			_, err := evaluate(raw)
			// Assert: a matching self-hash alone cannot authorize a different profile.
			if !errors.Is(err, errInvalid) {
				t.Fatal(err)
			}
		})
	}
}

func TestCaptureConfigurationMatchesActualRecipeLimits(t *testing.T) {
	// Arrange: the actual constructor config used by each capture strategy.
	recorded := referenceConfiguration()
	for i, profile := range profiles()[1:] {
		// Act: inspect configuration without dispatching host ports.
		actual := comparisonRecipeConfig(nil, liveModelPorts{}, recipe.Strategy(profile), new(observation))
		// Assert: recorded recipe controls match executable controls.
		if actual.MaxQueries != recorded.QueryLimits[i+1] || actual.Limits.ModelCalls != recorded.ModelLimits[i+1] ||
			actual.Limits.RetrievalCalls != recorded.RetrievalLimits[i+1] || actual.Duration.Nanoseconds() != recorded.AttemptNanos ||
			actual.Limits.Usage.InputTokens != recorded.InputCap || actual.Limits.Usage.OutputTokens != recorded.OutputCap ||
			actual.Limits.Usage.Cost != recorded.CostCap || actual.MaxDocuments != recorded.MaxDocuments || actual.FusionK != recorded.FusionK {
			t.Fatal(profile, actual, recorded)
		}
	}
}

func TestCodexCaptureIdentityBindsHostProfile(t *testing.T) {
	// Arrange: calibrated advisory controls, retaining the full observed grid.
	raw := comparisonCapture(t)
	raw.Configuration = cliConfiguration()
	raw.HostProfile = &codexHostProfile{
		Kind:       "codex-cli-calibrated",
		CLIVersion: "calibrated-cli",
		Model:      raw.ModelIdentity,
		CalibrationSHA256: digestConfigurationBytes(
			[]byte("calibration"),
		),
		AttemptNanos:  int64(cliAttemptDuration),
		CallNanos:     int64(calibrationCallDuration),
		TokenPolicy:   "advisory-reported-usage",
		PricePolicy:   "advisory-unknown-provider-price",
		ToolIsolation: "disabled-supported-tools-reject-observed-tools-additional-context-unverified",
	}
	raw.ConfigIdentity = captureConfigurationIdentity(raw)
	// Act: a retained identity cannot authorize a different model executor version.
	_, validErr := evaluate(raw)
	raw.HostProfile.CLIVersion = "changed-after-run"
	_, changedErr := evaluate(raw)
	// Assert.
	if validErr != nil || !errors.Is(changedErr, errInvalid) {
		t.Fatal(validErr, changedErr)
	}
}

func TestCodexRecipeControlsPreserveCallCapsWithAdvisoryPrice(t *testing.T) {
	// Arrange: the same constructors used by real capture, without dispatching ports.
	controls := cliConfiguration()
	for i, profile := range profiles()[1:] {
		actual := comparisonRecipeConfig(nil, liveModelPorts{}, recipe.Strategy(profile), new(observation))
		// Act.
		configureCodexRecipe(&actual, profile)
		// Assert: only explicit host duration/reservations/pricing differ.
		if actual.RequireKnownCost || actual.Duration.Nanoseconds() != controls.AttemptNanos ||
			actual.Limits.ModelCalls != controls.ModelLimits[i+1] || actual.Limits.RetrievalCalls != controls.RetrievalLimits[i+1] ||
			actual.Limits.Usage.InputTokens != controls.InputCap || actual.Limits.Usage.OutputTokens != controls.OutputCap {
			t.Fatal(profile, actual, controls)
		}
	}
}
