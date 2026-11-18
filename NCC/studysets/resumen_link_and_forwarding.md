# Links Switches and Routers

Full explanations from lecture-note sections 10, 11. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

## 10 Ethernet Wi Fi and the physical medium

### Ethernet frames and local delivery

In untagged Ethernet II, a frame contains a 6-byte destination MAC address, a 6-byte source MAC address, a 2-byte EtherType, a payload, and a 4-byte frame check sequence. EtherType 0x0800 identifies IPv4, 0x86DD identifies IPv6, and 0x0806 identifies ARP. The ordinary Ethernet payload range is 46 to 1500 bytes, with padding if necessary. Preamble and start-frame delimiter precede the MAC frame on the medium, and an interframe gap separates transmissions.

Ethernet’s frame check sequence uses a cyclic redundancy check to detect accidental corruption. A receiver can discard a bad frame. Conventional Ethernet does not provide the same per-frame acknowledgment and retry mechanism used for ordinary unicast Wi-Fi delivery. TCP’s end-to-end reliability therefore remains necessary even when a particular local link has error detection or local retries.

On a modern switched full-duplex Ethernet link, each direction can send without the shared-medium collision behavior of old half-duplex Ethernet. CSMA/CD belongs to that older shared-medium setting; describing a full-duplex switch port as routinely waiting for collisions would be misleading. A switch still has finite output capacity and buffers, so congestion and drops remain possible.

### Wi Fi changes local delivery

Suppose Alice’s device uses an infrastructure Wi-Fi LAN. It sends an 802.11 data frame to an access point. In the ordinary three-address station-to-distribution-system case, address 1 is the receiver and BSSID of the access point, address 2 is the transmitting station, and address 3 identifies the destination on the distribution system. If the IP next hop is a separate gateway, the AP’s radio receiver address and that gateway’s link destination can therefore be different.

The AP can bridge the payload to Ethernet. This changes the link framing while preserving the carried IP packet in an ordinary bridging operation. An AP that also performs routing is acting in an additional role. Wi-Fi is not a reason for the SMTP server to use a different application protocol or for TCP to stop using sequence numbers.

Wi-Fi uses carrier sensing, collision avoidance, randomized backoff, and local acknowledgments for common unicast traffic. A sender cannot reliably detect a radio collision while transmitting in the same manner as classic wired CSMA/CD. Hidden terminals and varying radio conditions complicate channel sharing. Local retransmissions can recover errors before TCP notices a loss, but they can increase delay and jitter. If retries fail, TCP may eventually recover the missing stream bytes.

The official wireless topic page lists 802.11 architecture, MAC operation, frame format, and higher-layer impact. At the time consulted, the chapter 7 entries used here link mainly to knowledge checks rather than a complete Wi-Fi video. Do not assume every item on the official site has an associated video.

### Signals carry the frame

The physical layer converts encoded data into electrical changes, light pulses, or modulated radio signals. It is concerned with transmission rate, signal quality, encoding, and propagation through the medium. The representation depends on the technology; saying that a wire carries abstract zeros and ones omits the signal encoding step. At the receiving interface, the physical implementation recovers symbols and bits for link-layer processing.

Transmission delay is the time required to place a specified number of bits onto a link. Propagation delay is the travel time through the medium. A faster link reduces transmission delay for a fixed frame, but it does not make light or radio propagate faster through a fixed path. Queueing adds variable delay before a busy output link becomes available.

## 11 Switches routers and a packet journey

### How an Ethernet switch learns

A conventional learning switch records the source MAC address of an arriving frame and its ingress port, within the relevant VLAN. For a known unicast destination, it forwards toward the learned destination port. For an unknown unicast destination, it floods eligible ports in that VLAN except the ingress port. Broadcast frames are similarly flooded within their broadcast domain. Entries age, allowing the table to adapt to movement.

S1 can learn A’s location from A’s ARP request and R1’s location from the reply. A’s later IPv4 frame has R1’s MAC destination, so S1 forwards it toward R1. It does not need to understand SMTP, and it does not create a route to bob@recipient.example. A normal bridge does not decrement the carried IP packet’s TTL. A VLAN can separate broadcast domains even on the same physical switch; routing is needed between IP subnets on different VLANs.

### What a router does to the frame and packet

R1 receives a frame addressed to its local interface. It validates the link-layer reception, removes that link encapsulation, examines the IP packet, and performs a forwarding lookup. It decreases TTL, updates the IPv4 checksum, determines the outgoing next hop, and constructs the outgoing link frame. If the neighbor mapping is absent, it must resolve it before transmitting in this Ethernet example. The router does not normally deliver the TCP segment to a local SMTP process when merely forwarding traffic to B.

The input Ethernet frame ends at R1. The output frame has R1’s transit MAC as source and R2’s transit MAC as destination. R2 repeats the forwarding process, then frames the packet with R2’s destination-LAN MAC as source and B’s MAC as destination. Each outgoing frame gets a frame check sequence for its contents. A switch along one bridged LAN generally preserves the source and destination MAC addresses as it forwards the frame.

| Observation point | Frame source | Frame destination | IP source to destination | TTL |
| --- | --- | --- | --- | --- |
| A toward R1 through S1 | A LAN MAC | R1 LAN MAC | 192.0.2.10 to 203.0.113.25 | 64 |
| R1 toward R2 | R1 transit MAC | R2 transit MAC | Same IP pair | 63 |
| R2 toward B through S2 | R2 LAN MAC | B LAN MAC | Same IP pair | 62 |

This table assumes no NAT, tunneling, fragmentation, or middlebox modification. The TCP source and destination ports remain 49152 and 25. The TCP sequence field and payload remain unchanged for this forwarded segment. The IPv4 checksum changes because TTL changes. The TCP checksum stays the same because TTL is not part of its pseudoheader.

### What the devices retain

The host’s DNS cache maps names and resource records to cached answers. Its neighbor cache maps local next-hop addresses to link addresses. A switch forwarding database maps MAC addresses to ports within VLANs. A router forwarding table maps IP prefixes to forwarding actions. TCP connection state tracks byte numbers, acknowledgments, windows, and timers. A mail queue stores application messages awaiting processing. These are different kinds of state; none can substitute for all the others.

![Figure 4  Different tables and buffers solve different delivery problems](figures/state.png)

Figure 4  Different tables and buffers solve different delivery problems

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Link layer and LANs](https://gaia.cs.umass.edu/kurose_ross/videos/6/)
- [Wireless and mobile networks](https://gaia.cs.umass.edu/kurose_ross/videos/7/)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
