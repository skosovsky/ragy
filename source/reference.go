// Package source defines immutable source artifact identities, independently of
// storage, domain metadata, authorization policy and parser/model engines.
package source

import (
	"fmt"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// Reference identifies one artifact in an exact source revision/representation.
// Revision, transformation and access fingerprints are host-provided identities;
// no latest alias or implicit representation conversion is supported.
type Reference struct {
	Namespace         string `json:"namespace"`
	Source            string `json:"source"`
	Revision          string `json:"revision"`
	Transformation    string `json:"transformation"`
	AccessFingerprint string `json:"access_fingerprint"`
	Artifact          string `json:"artifact"`
	Representation    string `json:"representation"`
}

// Validate rejects incomplete or non-roundtrippable identities without disclosing
// their values in error text. Reference is comparable and contains no mutable data.
func (r Reference) Validate() error {
	for _, value := range []string{r.Namespace, r.Source, r.Revision, r.Transformation, r.AccessFingerprint, r.Artifact, r.Representation} {
		if value == "" || !utf8.ValidString(value) {
			return fmt.Errorf("%w: source artifact reference", ragy.ErrInvalidArgument)
		}
	}
	return nil
}
