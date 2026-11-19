# Feature definitions and preparation rules

Feature schema version: 1. All features are derived from trimmed URL text.
No numerical columns supplied by the publisher are used as model inputs.
Character lengths and entropy operate on Unicode code points, not encoded bytes.
ASCII digits are 0–9; special characters are all non-alphanumeric code points.
Hostnames are lowercased, converted to IDNA ASCII, and stripped of a trailing dot.

| Feature | Definition |
| --- | --- |
| url_length | Characters in the original trimmed URL, including any scheme. |
| hostname_length | Characters in the parsed, normalized hostname, excluding port/userinfo. |
| path_length | Characters in the parsed path, including slashes. |
| query_length | Characters in the query, excluding `?`. |
| subdomain_count | Labels before the registrable domain, using the bundled Public Suffix List. |
| path_component_count | Nonempty path components separated by `/`. |
| digit_count | ASCII digits in the entire trimmed URL. |
| digit_ratio | digit_count / url_length. |
| special_character_count | Non-alphanumeric characters in the entire trimmed URL. |
| special_character_ratio | special_character_count / url_length. |
| dot_count | Literal `.` characters in the entire trimmed URL. |
| hyphen_count | Literal `-` characters in the entire trimmed URL. |
| at_count | Literal `@` characters in the entire trimmed URL. |
| percent_count | Literal `%` characters in the entire trimmed URL. |
| query_parameter_count | Nonempty components of the query separated by `&`. |
| is_https | 1 only if the original trimmed string begins with `https://`, case-insensitively. |
| hostname_is_ip | 1 for a standard IP address accepted by Python ipaddress. |
| has_explicit_port | 1 when a valid port is explicitly present. |
| url_entropy | Shannon entropy in bits per character over the entire trimmed URL. |
| hostname_entropy | Shannon entropy over the normalized hostname. |
| has_userinfo | 1 when the parsed authority contains `@`. |
| fragment_length | Characters after the fragment separator `#`. |

URLs without a scheme are parsed with an assumed HTTP prefix only for extracting
components; the prefix is not added to URL text features. Their HTTPS flag is 0.
Protocol-relative URLs beginning `//` are also allowed. The original URL is kept.
Only HTTP/HTTPS-style URLs are accepted. Empty URLs, whitespace/control characters,
malformed hostnames, unrecognized public suffixes, and invalid ports are excluded
with source-row reasons. Standard IPv4/IPv6 hosts are allowed and grouped by IP.
Unknown suffix exclusions may affect coverage and must be reported.

Labels must be numeric 0/1 (0=legitimate, 1=phishing). Missing labels are excluded.
Unexpected nonempty labels stop the run rather than being guessed. Missing-value
profiling counts empty/whitespace cells in every source column. Literal values
such as `NA` are not silently converted to missing, because they may be URL text.
Source numerical columns are profiled for blanks but not imputed or used.

Duplicate keys normalize scheme, hostname, IDNA, default ports, and an empty path
to `/`. Userinfo, path case, query order, and fragment are preserved. Duplicate
canonical URLs with the same label retain the first source row. All observations
of a canonical URL with conflicting labels are quarantined. Reasons are mutually
exclusive, and retained plus excluded rows must equal the input total.

Splitting groups standard hostnames by registrable domain, using the fixed
`tldextract==5.3.0` package's bundled PSL snapshot with no HTTP fetching. Private
suffixes are not treated as separate public suffixes, so e.g. blogspot subdomains
stay in one conservative group. This may reduce the number of groups on shared
hosting. Scheme, port, and subdomain variants of a registrable domain cannot cross
the split. No exact canonical URL may cross the split either.

No URL, hostname string, domain identity, source-row identifier, or label is a
model feature. Logistic Regression standardization is fitted only on training
features. Every model uses the same training and held-out observations.
