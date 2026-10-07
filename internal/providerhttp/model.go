package providerhttp

import (
	"encoding/json"

	ragy "github.com/skosovsky/ragy"
)

// ModelEcho decodes an optional/null string echo without asserting remote identity.
// Adapters compare a supplied value with their declared space; absence is not attestation.
func ModelEcho(raw json.RawMessage) (string, bool, error) {
	if len(raw) == 0 {
		return "", false, nil
	}
	var model *string
	if err := json.Unmarshal(raw, &model); err != nil {
		return "", false, ragy.ErrProtocol
	}
	if model == nil {
		return "", false, nil
	}
	return *model, true, nil
}
