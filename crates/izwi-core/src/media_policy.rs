//! Which network addresses remote media may be fetched from: public unicast
//! only. Shared by Qwen3.5's image fetches and the speech API's reference
//! audio URLs, so both refuse loopback, private, link-local, documentation
//! and other special-purpose addresses the same way (SSRF).

use std::net::IpAddr;

/// Whether `address` is publicly routable unicast.
pub fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_media_ipv4(address.octets()),
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return is_public_media_ipv4(mapped.octets());
            }
            is_public_media_ipv6(address.segments())
        }
    }
}

#[allow(clippy::nonminimal_bool)]
fn is_public_media_ipv4([first, second, third, _fourth]: [u8; 4]) -> bool {
    if first == 0
        || first == 10
        || first == 127
        || first >= 224
        || (first == 100 && (64..=127).contains(&second))
        || (first == 169 && second == 254)
        || (first == 172 && (16..=31).contains(&second))
        || (first == 192 && second == 168)
    {
        return false;
    }

    // IANA special-purpose, documentation, deprecated relay, and benchmark
    // networks are never valid media origins even when an OS route exists.
    if (first == 192 && second == 0 && third == 0)
        || (first == 192 && second == 0 && third == 2)
        || (first == 192 && second == 88 && third == 99)
        || (first == 198 && matches!(second, 18 | 19))
        || (first == 198 && second == 51 && third == 100)
        || (first == 203 && second == 0 && third == 113)
    {
        return false;
    }

    true
}

#[allow(clippy::nonminimal_bool)]
fn is_public_media_ipv6(segments: [u16; 8]) -> bool {
    let first = segments[0];
    let second = segments[1];

    // Public unicast is currently allocated from 2000::/3. This excludes
    // unspecified, loopback, ULA, link-local, multicast, IPv4-compatible, and
    // NAT64 literals without relying on unstable standard-library predicates.
    if first & 0xe000 != 0x2000 {
        return false;
    }

    // Conservatively reject the special-purpose portion of 2001::/23,
    // documentation prefixes, deprecated 6to4, and the 3fff::/20
    // documentation block.
    if (first == 0x2001 && second <= 0x01ff)
        || (first == 0x2001 && second == 0x0db8)
        || first == 0x2002
        || first & 0xfff0 == 0x3ff0
    {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::is_public_ip;

    #[test]
    fn only_public_unicast_passes() {
        for address in ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"] {
            assert!(is_public_ip(address.parse().unwrap()), "{address}");
        }
        for address in [
            "127.0.0.1",
            "10.1.2.3",
            "192.168.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "fd00::1",
            "::ffff:127.0.0.1",
            "2001:db8::1",
        ] {
            assert!(!is_public_ip(address.parse().unwrap()), "{address}");
        }
    }
}
