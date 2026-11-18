# IP Routing and Neighbor Resolution

Full explanations from lecture-note sections 8, 9. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

## 8 IP addressing routing and forwarding

### IP provides the packet service between networks

IP carries datagrams toward destination addresses using best-effort delivery. Packets can be lost, reordered, duplicated, or delayed. IP does not supply TCP’s connection establishment or ordered stream. A router forwards packets according to its forwarding table; routing protocols and configuration create the information used to build that table.

At A, the operating system compares the destination with its routing table. A’s directly connected route is 192.0.2.0/24. B’s address 203.0.113.25 does not match that prefix. A therefore uses a route through 192.0.2.1, normally its default gateway in this example. The packet destination remains B; the first local frame is addressed to R1. A does not replace the destination IP address with its gateway’s address.

Prefix length /24 means that the first 24 bits identify the network prefix. The corresponding mask is 255.255.255.0. For the transit /30, the mask is 255.255.255.252 and the example usable interface addresses are .1 and .2. Subnetting determines address and route relationships; it is independent of the text of the SMTP recipient address.

### Longest prefix match

When several routes match, forwarding selects the route with the longest matching prefix. Imagine a router has these entries:

| Destination prefix | Action |
| --- | --- |
| 0.0.0.0/0 | Forward toward upstream |
| 203.0.0.0/16 | Forward toward R7 |
| 203.0.113.0/24 | Forward toward R2 |
| 203.0.113.128/25 | Forward toward R8 |

For 203.0.113.25, the /24 beats the /16 and default. The /25 does not match because it covers .128 through .255. For 203.0.113.200, the /25 wins. Longest prefix means the most specific matching prefix, not the numerically largest destination address or the smallest number of router hops.

Forwarding is the local decision that maps an arriving packet to an outgoing interface and next hop. Routing is the broader process of discovering or choosing paths. A packet does not run a complete routing-protocol calculation at every hop. Typical control-plane topics include OSPF within an administrative domain and BGP between domains; these help supply routes, while the data plane applies forwarding entries to individual packets.

### IPv4 and IPv6 headers

An ordinary IPv4 header contains version, header length, differentiated-services information, total length, identification and fragmentation fields, TTL, Protocol, header checksum, and source and destination addresses. Version is 4; an option-free header has IHL 5, meaning 20 bytes. Protocol 6 identifies TCP. Total Length includes the IPv4 header and payload. The IPv4 header checksum protects the header, not the SMTP data.

A forwarding router decreases TTL and updates the IPv4 header checksum. If TTL expires, the packet is discarded and an ICMP Time Exceeded response may be returned. This limits persistent forwarding loops. It is not an email expiration timer. ICMP also carries other network diagnostics and errors; ICMP messages are not TCP acknowledgments.

IPv6 has 128-bit addresses and a fixed 40-byte base header. Its corresponding fields include Next Header and Hop Limit. It has no base-header checksum. IPv6 routers do not fragment packets; a source must adapt to path MTU information or perform permitted source fragmentation. For IPv4, a router may fragment when allowed, while DF can require discarding an oversized packet and returning an appropriate ICMP error. Avoiding fragmentation is generally preferable for TCP.

The packet examples use option-free IPv4 and no fragmentation. IPv4 field definitions are in RFC 791; IPv6’s base behavior is in RFC 8200. Modern implementations also follow updates to these foundational documents.

### NAT changes the simple address invariance

In the base routed example, IP source and destination remain A and B across routers. A typical home network adds address-and-port translation. A private client 10.0.0.10:49152 might appear externally as 192.0.2.50:62001 when contacting B:587. The NAT maintains a mapping and rewrites the return traffic toward the private endpoint. It must adjust the affected checksums. Therefore, claims that IP addresses and ports never change in transit need an explicit no-NAT assumption.

NAT is not mail relay. It translates packet headers without becoming the SMTP endpoint in this example. A mail relay instead terminates the application transaction and originates another one. A firewall is another distinct function: it can permit or block traffic according to policy, sometimes with connection state. A failed connection can result from policy even when the DNS answer and IP route are correct.

## 9 ARP and IPv6 Neighbor Discovery

### Resolve the next hop rather than the remote mail host

A has selected R1 as the next hop. If A lacks a mapping for 192.0.2.1, it broadcasts an ARP request on its LAN. The Ethernet destination is ff:ff:ff:ff:ff:ff. The request asks which interface owns 192.0.2.1 and includes A’s sender address information. R1 normally replies directly with its LAN MAC address. A can cache the mapping and use it for subsequent frames.

```text
Ethernet ARP request
Source MAC:      02:00:00:00:01:10
Destination MAC: ff:ff:ff:ff:ff:ff
EtherType:       0x0806
Sender IPv4:     192.0.2.10
Target IPv4:     192.0.2.1

Learned neighbor entry
192.0.2.1 -> 02:00:00:00:01:01
```

ARP does not ask DNS for a MAC address. ARP is carried directly in link frames, not inside IPv4, TCP, or UDP. Ordinary routers do not forward the LAN’s ARP broadcasts into the next IP subnet. R1 performs a separate resolution on its transit link if needed, and R2 resolves B on the final LAN. If B were on A’s own local subnet, A would normally resolve B directly instead of sending to the default gateway. RFC 826 defines the basic exchange.

### Neighbor Discovery in IPv6

IPv6 uses Neighbor Discovery, carried in ICMPv6, instead of ARP. Neighbor Solicitation and Neighbor Advertisement can resolve a neighboring IPv6 address to a link-layer address. Initial address-resolution solicitations use an appropriate solicited-node multicast destination rather than an Ethernet broadcast. Neighbor Discovery also supports router discovery and neighbor reachability tracking, so it is broader than address resolution alone.

A router advertisement can provide default-router and on-link-prefix information. DHCPv6 can provide other configuration; learning a default router is not simply identical to IPv4 DHCP operation. These distinctions matter when describing how a host knows where to send an off-link packet. RFC 4861 supplies the Neighbor Discovery framework. On a routed IPv6 path, the host still resolves a local next hop, not the remote server’s MAC address across the Internet.

### Configuration before the running example

An IPv4 host may obtain an address, mask, default gateway, and resolver information using DHCP. In an initial IPv4 DHCP exchange, a client can use broadcasts before it has a usable address, and a relay can help reach a server outside the LAN. Static configuration is another possibility. A Wi-Fi host must also join the wireless network and complete relevant access security before sending ordinary application traffic. These are prerequisites, not operations repeated for each SMTP command.

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Network data plane](https://gaia.cs.umass.edu/kurose_ross/videos/4/)
- [Neighbor Discovery](https://www.rfc-editor.org/rfc/rfc4861)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
