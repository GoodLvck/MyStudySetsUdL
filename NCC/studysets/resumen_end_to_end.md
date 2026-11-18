# Packet Walkthrough and Worked Examples

Full explanations from lecture-note sections 12, 13, 14, 15, 16. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

## 12 Encapsulation and a fully specified packet

### A small EHLO command with explicit byte counts

Use the TCP handshake from section 7, then let B send this exact 39-byte greeting, including its CRLF:

```text
220 mx1.recipient.example ESMTP ready\r\n
```

B’s greeting starts at sequence 7001. Its next sequence position is 7040. A can acknowledge 7040 while sending the exact 26-byte command below. Assume no other payload has been exchanged, and no TLS or TCP options are used in this teaching packet.

```text
EHLO mail.sender.example\r\n

Payload in hexadecimal
45 48 4c 4f 20 6d 61 69 6c 2e 73 65 6e
64 65 72 2e 65 78 61 6d 70 6c 65 0d 0a
```

The literal backslash notation shows the two control bytes 0d and 0a; the backslash letters are not transmitted. There are 24 printable bytes plus 2 line-ending bytes. The packet is not an entire email. It carries one SMTP command early in the transaction.

| Layer or field | Value in this example |
| --- | --- |
| Ethernet destination | 02:00:00:00:01:01 |
| Ethernet source | 02:00:00:00:01:10 |
| EtherType | 0x0800 for IPv4 |
| IPv4 Version and IHL | 4 and 5, a 20 byte header |
| IPv4 Total Length | 66 bytes, equal to 20 + 20 + 26 |
| IPv4 Identification | 0x1a2b |
| IPv4 flags and offset | DF set, offset zero |
| IPv4 TTL and Protocol | 64 and 6 |
| IPv4 source and destination | 192.0.2.10 and 203.0.113.25 |
| TCP source and destination ports | 49152 and 25 |
| TCP sequence and ACK | 1001 and 7040 |
| TCP data offset and flags | 5, ACK set, no other flags |
| TCP receive window | 64240, no scaling in this example |
| TCP payload | 26 bytes |

The Ethernet MAC frame is 14 + 66 + 4 = 84 bytes, including its FCS. It needs no padding because the Ethernet payload is already larger than 46 bytes. On the medium, the preamble and start delimiter add 8 byte-times; the interframe gap adds another 12 byte-times of spacing. The resulting occupied interval is 104 byte-times under these ordinary Ethernet assumptions. Neither the preamble nor the gap belongs to IPv4 Total Length.

### The corresponding headers in hexadecimal

The following header values are computed for this exact example. Checksums are shown separately so they can be recognized rather than mistaken for payload. Network byte order places the most significant byte first. The Ethernet FCS and physical preamble are omitted from this header display.

```text
Ethernet header
02 00 00 00 01 01  02 00 00 00 01 10  08 00

IPv4 header
45 00 00 42 1a 2b 40 00 40 06 22 67 c0 00 02 0a cb 00 71 19

TCP header
c0 00 00 19 00 00 03 e9 00 00 1b 80 50 10 fa f0 82 a3 00 00
```

The TCP source port 49152 appears as c0 00 and destination port 25 as 00 19. Sequence 1001 is 00 00 03 e9; ACK 7040 is 00 00 1b 80. The TCP checksum calculation includes a pseudoheader with the IP source, IP destination, protocol, and TCP length. The pseudoheader is input to checksum calculation; it is not an extra transmitted header between IP and TCP.

At the far end, B can acknowledge client sequence 1027 after receiving all 26 EHLO bytes. That ACK can be carried with the SMTP EHLO reply rather than in a separate empty segment. A particular packet capture can use different segmentation and ACK timing from this simplified example without violating the protocol.

### Captures can differ from wire packets

An endpoint capture can display incorrect-looking TCP checksums when checksum calculation is deferred to the interface. Segmentation offload can also make captured buffers appear larger than wire packets, while receive offload can coalesce data. Many captures omit the Ethernet FCS, and an ordinary host capture on Wi-Fi may expose an Ethernet-like view rather than raw radio frames. Therefore a displayed packet length must be interpreted with the capture location and format in mind. These notes’ wire calculations assume the explicit ordinary frames described above.

## 13 Receiving and decapsulation

### From signals to a socket

B’s interface recovers the incoming frame from the medium. Link-layer processing checks that the frame is relevant to the interface and validates its error-detection information. The EtherType directs the payload to IPv4 processing. IPv4 validates the header, recognizes B as the destination, and passes a Protocol 6 payload to TCP after any necessary IP reassembly. Unlike a forwarding router, B is the IP destination and does not forward this packet onward toward itself.

TCP validates the segment, finds the connection from its address and port information, and interprets sequence and acknowledgment fields. It buffers and orders data and makes contiguous stream bytes available through the connected socket. The SMTP process reads those bytes and parses complete CRLF-terminated protocol lines according to its state. Removing a TCP header does not automatically yield one complete SMTP command; a command may require bytes from several segments.

For the DATA phase, the SMTP process switches from command parsing to receiving message content until the SMTP terminator is recognized. It removes dot transparency, applies its acceptance policy, and hands accepted content to the mail system’s processing and storage path. The system can add trace headers such as Received. The final response states application acceptance. If B is a relay rather than the mailbox’s final system, another transfer may follow.

### The reply makes a full return journey

When B replies, its application bytes go down its own stack. TCP uses source port 25 and destination port 49152 for the same connection. The IP source is B and destination is A. B chooses a route toward A and normally sends its first frame to its local gateway, R2. Neighbor resolution and frame construction take place on B’s local link. Routing can send the return packets along a different path, so the sequence of reverse routers need not be a mirror image.

Every SMTP reply follows the same general stack path as a command. It is easy to describe only the outbound message, but the greeting, capabilities, acceptance, transport acknowledgments, and errors all depend on reverse-direction communication. A one-way path failure can therefore prevent progress even if packets appear to leave the sender successfully.

## 14 Worked examples of size timing and loss

### Segmentation of a larger byte stream

Assume 5000 bytes of application data are ready for TCP, a path MTU of 1500 bytes, a 20-byte IPv4 header, and a 20-byte TCP header with no options. Assume MSS and sender behavior permit full-sized segments. The maximum payload under these assumptions is 1500 − 20 − 20 = 1460 bytes. The 5000 bytes can occupy payloads of 1460, 1460, 1460, and 620 bytes. This is TCP segmentation, not IPv4 fragmentation.

The first three IP packets are 1500 bytes each; the fourth is 660 bytes. The IP bytes total 5160, including four IP and four TCP headers. With untagged Ethernet, add 18 bytes per frame for the Ethernet header and FCS: 5232 MAC-frame bytes in total. With 8 preamble bytes and 12 gap byte-times per frame, the occupied transmission intervals total 5312 byte-times. This calculation excludes connection setup, DNS, ARP, ACK-only packets, SMTP replies, TLS, and all other message bytes. It measures a specific data-transfer slice rather than the total cost of delivering an email.

If TCP timestamps add 12 bytes to each TCP header, the assumed 1460-byte payload no longer fits under the same 1500-byte IP MTU. Payload sizing must account for actual headers. For IPv6 without extensions, the corresponding base calculation is 1500 − 40 − 20 = 1440 bytes. The peer’s MSS and the discovered path MTU can impose further limits.

### A byte numbering and loss example

Choose a separate illustrative client stream starting at sequence 1001, with 1460-byte data segments. Segment 1 covers 1001–2460. Segment 2 covers 2461–3920. Segment 3 covers 3921–5380. If segment 2 is lost but segment 3 arrives, the receiver’s cumulative ACK remains 2461 because that is the first missing byte. It may buffer segment 3 and report its range through SACK if negotiated.

Once the missing range arrives, the cumulative ACK can advance to 5381. The application can then receive the previously blocked contiguous bytes. This example does not claim that one later segment alone triggers classic fast retransmit: that mechanism commonly requires three duplicate ACKs. Additional later segments could supply those ACKs; otherwise timer-based or other implemented loss recovery may be needed.

### Transmission propagation and queueing

A 1500-byte IP packet contains 12000 bits. At 100 Mb/s, serializing those IP bits takes 120 microseconds. At 1 Gb/s, it takes 12 microseconds. These figures exclude Ethernet overhead. For a full 1500-byte Ethernet payload, the occupied interval including 18 MAC-frame overhead bytes, 8 preamble bytes, and 12 gap byte-times is 1538 × 8 / 100000000 = 123.04 microseconds at 100 Mb/s.

If a link spans 2000 km and signals propagate at 2 × 10^8 m/s, propagation delay is 2000000 / 200000000 = 0.01 seconds, or 10 ms. The frame’s bits do not wait until serialization completes before the first bit begins propagating. For one store-and-forward hop, a useful model adds processing, queueing, transmission, and propagation. Across a path, add the relevant delays for all links and devices under the assumptions being used.

Queueing depends on the competing load and can dominate variable delay. A tenfold link-rate increase reduces serialization time but does not eliminate a 10 ms propagation component. TCP retransmission and SMTP retry can add delays far larger than one frame transmission.

### Application round trips

Assume an already resolved destination, available neighbor entries, an RTT of 40 ms, negligible serialization and processing, plaintext SMTP, no pipelining, and a server that sends its greeting after receiving the final handshake ACK. Measure from the initial SYN. The client receives SYN ACK after one RTT and sends the final ACK. That ACK reaches the server half an RTT later, and the server greeting reaches the client after another half RTT. Thus the client has the greeting at about two RTTs from the initial SYN. EHLO and its reply cost another RTT, MAIL and its reply another, RCPT and its reply another, and DATA and 354 another. Sending a short complete content block and receiving the final 250 costs one more RTT.

Under these stated assumptions, application acceptance arrives at roughly 7 × 40 = 280 ms from the initial SYN. QUIT adds another request/reply interval if included. DNS resolution, a delayed greeting, TLS negotiation, authentication, multiple recipients, serialization of a large body, queueing, and losses increase this value. PIPELINING, when advertised and used under its rules, can reduce some waiting. The number 280 ms is a worked scenario, not a universal SMTP latency.

### Attachment expansion

Base64 represents each complete three-byte group using four encoded characters. For a 3000-byte attachment, that gives 4000 base64 characters before line wrapping. With 76-character lines, 53 lines are needed. If each line has a CRLF, that adds 106 bytes, producing 4106 bytes for the encoded content lines under this convention. MIME part headers, multipart boundaries, the rest of the message, and SMTP framing add further bytes. This explains why an attachment’s filesystem size is smaller than its transmitted mail representation.

## 15 Failure analysis and explained cases

### A correct name does not establish reachability

If the MX and address lookups succeed but A cannot resolve its gateway’s MAC address, the fault is local-link reachability or configuration. No SMTP greeting can arrive because the relevant IP traffic cannot leave the LAN. If ARP succeeds but SYNs receive no response, possibilities include filtering, a broken path, an unavailable destination, or missing return reachability. Silence alone does not uniquely identify the failing component.

If a TCP reset is returned to the initial SYN, the destination or a device on the path is actively refusing the attempt; an absent listener is a common explanation. If TCP establishes but the application sends a 550 recipient rejection, the network and TCP path have already carried useful application data. The failure is now an SMTP recipient or policy outcome, not proof that IP routing failed.

### A large transfer stalls while small exchanges work

Suppose greeting and EHLO succeed but a large DATA transfer repeatedly stalls. An MTU mismatch combined with blocked path-MTU feedback is one possible cause. Small packets fit, while larger packets with DF set cannot traverse a smaller outgoing MTU. Congestion, a middlebox, application policy, and endpoint failure are also possible. Packet sizes, retransmissions, ICMP messages, and SMTP logs provide evidence for distinguishing them.

A small successful ping does not prove that a large TCP transfer will succeed. Conversely, an unanswered ping does not prove that SMTP is unreachable because ICMP echo can be filtered separately. Interpret tests according to the protocol behavior they actually exercise.

### TCP success and SMTP success answer different questions

Suppose the client’s final DATA bytes are cumulatively acknowledged by TCP but the SMTP process rejects the message after scanning it. TCP has done its job. The application has made a separate decision. Suppose instead that the server queues the message and sends 250, but the reply is lost and the connection later fails. The server may own an accepted copy while the sender is unsure. Application-level retries must handle that uncertainty.

Another case is a temporary 451 after content. The client should normally retain the queued message for retry according to its delivery policy. A permanent failure for one recipient is handled differently. An SMTP acceptance can also be followed later by a delivery status notification if downstream delivery fails. No one transport flag proves that a person read a message.

### Same LAN and remote LAN reasoning

If A sends to a peer within 192.0.2.0/24 through the same switched VLAN, it uses the peer’s MAC address after local resolution. The gateway is normally not in that packet’s forwarding path. S1 can deliver the frame without decrementing IP TTL. If A sends to B’s remote subnet, the first frame instead targets R1. The SMTP recipient, TCP port, and destination IP do not directly tell the Ethernet switch which output port to use; its learned MAC table supplies that decision.

### Reading an observed trace as evidence

A useful trace explanation follows actual protocol dependencies: identify the capture interface and direction; interpret visible DNS or cached resolution; identify the TCP connection tuple and handshake; follow stream byte numbers; interpret SMTP only when plaintext is available; compare local MAC destinations with IP destinations; and identify what the capture cannot show. A capture on A’s LAN does not directly show the MAC addresses used on B’s LAN. A missing ARP request may simply indicate a warm cache.

For encrypted SMTP, use endpoint application logs or a properly authorized lab decryption setup to interpret message-level events. Ciphertext alone cannot reveal a specific SMTP reply. Do not infer that DATA was accepted merely because the TLS connection carried bytes and ended normally.

## 16 A complete spoken explanation

An SMTP-sending process first needs a destination server. For submission it commonly uses a configured service; for direct relay the sending MTA resolves the recipient domain’s MX records and then resolves a selected exchanger to an IP address. These DNS exchanges are separate application traffic with their own lower-layer delivery.

The process creates a TCP socket and connects from an ephemeral port to the server’s SMTP port. The host chooses a route to the server IP. If the server is remote, the local next hop is a router. IPv4 ARP or IPv6 Neighbor Discovery supplies that next hop’s link address if the mapping is not cached. TCP’s three-way handshake establishes the byte-stream connection.

SMTP runs over that stream. The server greets the client, the client sends EHLO, and the peers can negotiate TLS and other extensions. The client supplies envelope sender and recipient information, obtains permission to send content, and transmits the message headers and body with the required framing. A final successful response after content indicates SMTP acceptance at that server.

For any one data segment, TCP adds source and destination ports, byte sequence and acknowledgment information, flags, window information, and a checksum. IP adds source and destination addresses and forwarding information such as TTL or Hop Limit. The local link layer wraps that packet in a frame addressed to the local next hop, and the physical layer transmits encoded signals.

A switch forwards within the local LAN using link addresses. A router removes the incoming link frame, consults the IP forwarding table, decreases the packet’s hop counter, and creates an outgoing frame for the next link. In a simple routed path without NAT, the endpoint IP addresses and TCP ports stay the same while link addresses change at router boundaries. The frames on Ethernet and Wi-Fi can have different formats while carrying the same network-layer packet.

At the destination host, physical and link processing recover and validate the frame. IP recognizes a local destination and dispatches the payload to TCP. TCP validates and reconstructs an ordered byte stream and supplies it to the correct socket. The SMTP process interprets complete protocol lines and content, then the mail system queues or delivers accepted mail. Its replies travel back through the stack. A relay repeats the application transfer using a new TCP connection; Bob’s later mailbox access is a separate conversation.

In OSI terms, this explanation covers application dialogue, representation and session functions, transport, network, data link, and physical delivery. In the Internet model, SMTP, DNS, MIME, and TLS functions sit above TCP; TCP sits above IP; IP uses the local link and physical mechanisms. The key is to explain both the layer responsibilities and the concrete transitions between them.

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Official lecture index](https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm)
- [TCP specification](https://www.rfc-editor.org/rfc/rfc9293)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
