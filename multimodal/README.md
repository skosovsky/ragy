# Multimodal transport-neutral parts

Text must be nonempty valid UTF-8. Inactive Text/URL/MIME fields must be literally
empty; whitespace is payload, not absence. Inactive Bytes permits nil/empty only.

URL parts accept an absolute URL after trimming for validation; Validate does not
rewrite the original value. This is a transport-neutral shape check, not a downloader
or SSRF policy. A custom absolute scheme may be valid here and unsupported by a
provider. Provider adapters enforce their supported schemes, MIME and wire formatting
(including treatment of surrounding whitespace). Host supplies network allowlists,
credentials and any fetching policy. No implicit normalization, fetch or fallback
is performed. Providers and host codecs must honor the declared input/space bounds.
