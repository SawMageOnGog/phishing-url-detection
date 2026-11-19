"""Extract lexical features without DNS, HTTP, or other network lookups."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit
import tldextract

# Fixed package version + bundled PSL snapshot; never fetch live suffix rules.
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                                include_psl_private_domains=False)
FEATURES = [
    'url_length', 'hostname_length', 'path_length', 'query_length',
    'subdomain_count', 'path_component_count', 'digit_count', 'digit_ratio',
    'special_character_count', 'special_character_ratio', 'dot_count',
    'hyphen_count', 'at_count', 'percent_count', 'query_parameter_count',
    'is_https', 'hostname_is_ip', 'has_explicit_port', 'url_entropy',
    'hostname_entropy', 'has_userinfo', 'fragment_length',
]

def entropy(value):
    """Shannon entropy in bits per Unicode character; empty string -> 0."""
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in Counter(value).values()) if n else 0.0

def parse_url(raw):
    value = str(raw).strip()
    if not value or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError('empty URL, whitespace, or control character')
    # Missing scheme is permitted, but is_https stays false unless https was present.
    if value.startswith('//'):
        parse_value = 'http:' + value
    elif '://' not in value:
        if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', value) and not re.match(r'^[^/:]+:\d+(?:/|$)', value):
            raise ValueError('unsupported or malformed scheme')
        parse_value = 'http://' + value
    else:
        parse_value = value
    parsed = urlsplit(parse_value)
    if parsed.scheme.lower() not in ('http', 'https'):
        raise ValueError('unsupported scheme')
    hostname = (parsed.hostname or '').rstrip('.').lower()
    if not hostname:
        raise ValueError('missing hostname')
    port = parsed.port  # Raises on malformed or out-of-range ports.
    try:
        hostname = ipaddress.ip_address(hostname).compressed
        is_ip = True
        domain_group = hostname
        subdomains = 0
    except ValueError:
        hostname = hostname.encode('idna').decode('ascii')
        labels = hostname.split('.')
        if len(hostname) > 253 or len(labels) < 2 or any(
            not label or len(label) > 63 or not re.fullmatch(r'[a-z0-9-]+', label)
            or label.startswith('-') or label.endswith('-') for label in labels
        ):
            raise ValueError('invalid domain hostname')
        ext = EXTRACTOR(hostname)
        if not ext.suffix or not ext.domain:
            raise ValueError('hostname has no recognized public suffix')
        domain_group = ext.top_domain_under_public_suffix
        subdomains = len(ext.subdomain.split('.')) if ext.subdomain else 0
        is_ip = False
    host_key = '[' + hostname + ']' if ':' in hostname else hostname
    if port is not None and (parsed.scheme.lower(), port) not in [('http', 80), ('https', 443)]:
        host_key += ':' + str(port)
    # Userinfo and path/query/fragment are preserved; host and scheme normalized.
    userinfo = parsed.netloc.rsplit('@', 1)[0] + '@' if '@' in parsed.netloc else ''
    canonical = parsed.scheme.lower() + '://' + userinfo + host_key + (parsed.path or '/')
    if parsed.query:
        canonical += '?' + parsed.query
    if parsed.fragment:
        canonical += '#' + parsed.fragment
    return value, parsed, hostname, bool(is_ip), domain_group, subdomains, port, canonical

def extract_features(raw):
    value, parsed, hostname, is_ip, group, subdomains, port, canonical = parse_url(raw)
    digits = sum('0' <= ch <= '9' for ch in value)
    special = sum(not ch.isalnum() for ch in value)
    result = dict(zip(FEATURES, [
        len(value), len(hostname), len(parsed.path), len(parsed.query),
        subdomains, len([x for x in parsed.path.split('/') if x]),
        digits, digits / len(value), special, special / len(value),
        value.count('.'), value.count('-'), value.count('@'), value.count('%'),
        len([x for x in parsed.query.split('&') if x]),
        int(value.lower().startswith('https://')), int(is_ip), int(port is not None),
        entropy(value), entropy(hostname), int('@' in parsed.netloc), len(parsed.fragment)
    ]))
    return result, {'domain_group': group, 'hostname': hostname, 'canonical_url': canonical}

def defang(value):
    return str(value).replace('https://', 'hxxps://').replace('http://', 'hxxp://').replace('.', '[.]')
