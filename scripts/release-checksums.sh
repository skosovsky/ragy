#!/bin/bash
# ragy adapters depend on the root module; retain its candidate sums.
set -euo pipefail
export GOWORK=off
checkout=${RELEASE_CANDIDATE_DIR:?}
: "${RELEASE_VERSION:?}" "${RELEASE_ARTIFACT_DIR:?}"
fail() { echo "release checksums: $*" >&2; exit 1; }
rootpath=$(cd "$checkout" && "${GO:-go}" list -m -f '{{.Path}}')
sumjson=$(cd "$RELEASE_ARTIFACT_DIR" && GOPROXY="file://$RELEASE_ARTIFACT_DIR" GONOSUMDB="$rootpath,$rootpath/*" "${GO:-go}" mod download -json "$rootpath@$RELEASE_VERSION")
sum=$(printf '%s\n' "$sumjson" | sed -n 's/^[[:space:]]*"Sum": "\([^"]*\)".*/\1/p')
modsum=$(printf '%s\n' "$sumjson" | sed -n 's/^[[:space:]]*"GoModSum": "\([^"]*\)".*/\1/p')
[[ "$sum" == h1:* && "$modsum" == h1:* ]] || fail 'missing candidate root checksums'
while IFS= read -r module; do
  [[ "$module" != . ]] || continue
  file="$checkout/$module/go.sum"
  [[ ! -L "$file" ]] || fail 'checksum file must not be a symlink'
  [[ -f "$file" ]] || : > "$file"
  awk -v p="$rootpath" -v v="$RELEASE_VERSION" '!($1==p && ($2==v || $2==v"/go.mod"))' "$file" > "$file.tmp"
  printf '%s %s %s\n%s %s/go.mod %s\n' "$rootpath" "$RELEASE_VERSION" "$sum" "$rootpath" "$RELEASE_VERSION" "$modsum" >> "$file.tmp"
  LC_ALL=C sort -u "$file.tmp" > "$file"; rm "$file.tmp"
done < "$checkout/scripts/release-modules.txt"
