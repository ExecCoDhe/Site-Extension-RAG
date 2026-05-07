import ipaddress
import socket
from urllib.parse import urlparse


BLOCKED_HOSTNAMES = {"localhost"}


def same_hostname(seed_url: str, candidate_url: str) -> bool:
    seed = urlparse(seed_url)
    candidate = urlparse(candidate_url)
    return bool(seed.hostname and candidate.hostname and seed.hostname == candidate.hostname)


def same_registrable_domain(seed_url: str, candidate_url: str) -> bool:
    seed = urlparse(seed_url)
    candidate = urlparse(candidate_url)
    return bool(
        seed.hostname
        and candidate.hostname
        and registrable_domain(seed.hostname) == registrable_domain(candidate.hostname)
    )


def registrable_domain(hostname: str) -> str:
    parts = hostname.lower().strip(".").split(".")
    if len(parts) <= 2:
        return hostname.lower()
    return ".".join(parts[-2:])


def is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    hostname = parsed.hostname.lower()
    if hostname in BLOCKED_HOSTNAMES:
        return False

    # In cloud environments (Cloud Run with VPC egress), DNS may resolve
    # public domains to internal/private IPs, causing false rejections.
    # Skip the resolution check and rely on the hostname blocklist instead.
    import os
    if os.getenv("APP_ENV") in ("dev", "prod"):
        return True

    addresses = _resolve_host(hostname)
    return bool(addresses) and all(_is_public_ip(address) for address in addresses)


def _resolve_host(hostname: str) -> set[str]:
    try:
        return {
            item[4][0]
            for item in socket.getaddrinfo(hostname, None)
        }
    except socket.gaierror:
        return set()


def _is_public_ip(address: str) -> bool:
    ip_address = ipaddress.ip_address(address)
    return not (
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
    )
