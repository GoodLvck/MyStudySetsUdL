# Complete Networking Lecture Notes

Detailed English lecture notes for the November 18, 2026 review. Follow one SMTP message through processes, sockets, DNS, TCP, IP, neighbors, and local links. The numerical traces state their assumptions explicitly.

[Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx) · [Topics](index.html)

## 1 The complete delivery story

### One email can require several network conversations

Alice composes a message addressed to bob@recipient.example. Her mail application is a running process. In a common desktop-mail configuration, it connects to a configured submission service, authenticates, and submits the message using SMTP. The service accepts responsibility for handling the message and places it in an outgoing queue. A sending mail transfer agent then finds the recipient domain’s mail server through DNS and opens another SMTP connection to that server. The receiving system accepts the message and passes it to local delivery machinery, which stores it in Bob’s mailbox. Bob later reads it through IMAP or a webmail service. SMTP transfers mail; mailbox access is a separate function.

If Alice uses webmail, her browser normally communicates with the webmail provider using HTTPS. An internal mail service then originates SMTP traffic. It would be inaccurate to claim that every email begins with an SMTP socket in the browser. The review’s process-to-process path can begin at whichever process actually originates the SMTP transaction.

There are therefore two useful scales. At the application scale, follow the email across submission, relay, and delivery. At the packet scale, select one TCP connection and follow its bytes through the stack. A TCP segment from the submission connection does not continue unchanged across a mail relay. The relay receives application data and later sends message data over a new connection with different ports, sequence numbers, and possibly different IP versions.

### The running network example

For the detailed packet examples, the sending MTA A is directly reachable at 192.0.2.10/24 and the receiving MTA B is at 203.0.113.25/24. A uses TCP source port 49152 and B listens on TCP port 25. The path is A, switch S1, router R1, router R2, switch S2, B. R1 connects the source LAN to a transit link; R2 connects that transit link to the destination LAN. All links in this base example are Ethernet. Wi-Fi and NAT are introduced separately so their effects remain explicit.

All domain names, addresses, and numbers in the examples are invented for teaching. The example IP ranges are reserved for documentation and should not be used as a live lab configuration on the public Internet. We assume interfaces, subnet masks, default routes, and DNS resolver information already exist; a later section explains how DHCP and IPv6 configuration fit before this starting point.

| Node or interface | IPv4 address | Ethernet address |
| --- | --- | --- |
| A | 192.0.2.10/24 | 02:00:00:00:01:10 |
| R1 source LAN | 192.0.2.1/24 | 02:00:00:00:01:01 |
| R1 transit | 198.51.100.1/30 | 02:00:00:00:02:01 |
| R2 transit | 198.51.100.2/30 | 02:00:00:00:02:02 |
| R2 destination LAN | 203.0.113.1/24 | 02:00:00:00:03:01 |
| B | 203.0.113.25/24 | 02:00:00:00:03:25 |

![Figure 1  The routed SMTP connection used in the packet examples](figures/topology.png)

Figure 1  The routed SMTP connection used in the packet examples

### What happens before the first email byte travels

The sending system resolves the destination mail service, selects a destination address, chooses a route, and resolves the local next hop’s link address if its neighbor cache lacks that information. TCP establishes a connection. Only then does the SMTP server greeting reach the sending application. DNS and ARP are supporting exchanges with their own messages; they are not extra headers attached to an SMTP message.

This order is a conceptual dependency, not a rigid universal packet trace. A cached DNS answer can remove a DNS exchange. A cached ARP entry can remove an ARP request. Resolving DNS can itself require the host to resolve its gateway’s MAC address first. Several activities can overlap, and the return path may differ from the forward path.

## 2 OSI and TCP IP models

### What a layer means

A protocol defines rules between peer entities, such as two SMTP processes or two TCP implementations. A service describes what a layer offers to the layer above it. An interface is the local boundary through which that service is used. The socket API is one such boundary. Logical peer communication does not mean that SMTP sends directly to remote SMTP without the lower layers. The implementation moves data down the local stack, across links, and up the remote stack.

The OSI model has seven layers. Internet teaching commonly uses five layers, while the four-layer TCP/IP model groups physical and data-link functions into a link or network-access layer. Both descriptions are useful if the grouping is stated. Neither requires a separate header for every conceptual function.

| OSI layer | Main function in this example | Five layer Internet model | Four layer TCP IP model |
| --- | --- | --- | --- |
| 7 Application | SMTP commands and replies; DNS queries | Application | Application |
| 6 Presentation | Encoding, representation, encryption functions | Application | Application |
| 5 Session | Application dialogue and session coordination | Application | Application |
| 4 Transport | TCP ports, ordered reliable byte stream | Transport | Transport |
| 3 Network | IP addressing and forwarding between networks | Network | Internet |
| 2 Data link | Ethernet or Wi-Fi local frames and MAC access | Link | Link or network access |
| 1 Physical | Electrical, optical, or radio signals | Physical | Link or network access |

### Where SMTP MIME and TLS belong

SMTP is an application-layer protocol. DNS is also an application-layer protocol, even though it supports many other applications. MIME describes message content types and encodings; it does not provide transport reliability. Its representation functions can be discussed using the OSI presentation concept, but Internet implementations do not require a separate OSI presentation protocol module for MIME.

TLS normally sits above TCP and below application plaintext. In the Internet five-layer teaching model it is commonly treated within application-layer functionality. It performs security functions reminiscent of the OSI presentation and session layers, but forcing it into exactly one OSI box can obscure the actual implementation. TCP itself provides a connection, but that does not make TCP an OSI session-layer protocol.

### Names for the data at each boundary

An SMTP command, reply, or message body becomes application bytes. TCP places a portion of its byte stream into a segment. IP carries the segment in a datagram or packet. Ethernet carries the IP packet in a frame and adds an error-detection trailer. The physical layer represents the frame using signals. Packet is also used informally for several of these units, so a precise explanation identifies the actual layer.

![Figure 2  Nested protocol units at a sending Ethernet interface](figures/encapsulation.png)

Figure 2  Nested protocol units at a sending Ethernet interface

Encapsulation does not promise a one-to-one mapping. One application write can become several segments. Multiple small writes can share a segment. One SMTP command can be split across segments, and multiple commands or replies can appear in one segment. TLS adds record boundaries that also need not match TCP segments. An SMTP parser must recognize its own delimiters after TCP has reconstructed the byte stream.

## 3 Processes sockets and addressing

### A process asks the operating system to communicate

A process is a program in execution, with memory and operating-system resources. A socket is an endpoint abstraction exposed through an API. In a typical TCP client, the application creates a socket, requests a connection, writes bytes, reads replies, and eventually closes the socket. The operating system normally implements TCP and IP. The network interface and its driver handle much of the link processing, with hardware offload possible.

The server creates a listening socket, binds it to a local address and service port, and listens. Accepting a connection produces a connected socket associated with that client. The listening socket remains available for additional clients. Several connected sockets can therefore have local port 25 while belonging to different connections.

The following is conceptual pseudocode. Real SMTP code needs reply parsing, multiline replies, timeouts, TLS handling, and correct error recovery.

```text
Client process                         Server process
socket(TCP)                            socket(TCP)
connect(B, 25)                         bind(local_address, 25)
                                       listen()
                                       accept() -> connected socket
read greeting                          write greeting
write command                          read and parse command
read reply                             write reply
close()                                close connected socket
```

A successful write usually means that the local system accepted bytes into a buffer. It does not prove that the remote TCP received them, that the SMTP process parsed them, or that Bob received a message. The application must interpret its own protocol replies. Reads may return fewer bytes than the sender wrote in one operation. An empty result from a stream read commonly indicates orderly end of stream, not an empty SMTP command.

### Four different kinds of identifier

The domain recipient.example identifies an administrative naming context. A DNS lookup helps locate a mail host. An IP address identifies a network-layer destination interface or addressable endpoint. A port identifies a transport endpoint within the host. A MAC address identifies a link-layer interface within the relevant link or bridged LAN. A mailbox local part, such as bob, is interpreted by the mail system and is not a TCP port.

For our connection, the identifying TCP tuple is (192.0.2.10, 49152, 203.0.113.25, 25). Adding the transport protocol gives the commonly used five-tuple. The reverse-direction segments use the reversed source and destination pairs. The sender’s ephemeral port is selected locally; the receiver’s service port is known by convention or configuration. Another client can use the same source port if its complete connection tuple is different.

Multiplexing lets data from many applications share the host’s network interfaces. Demultiplexing directs arriving data to the right protocol and socket. Ethernet’s EtherType can identify IPv4; IPv4’s Protocol field can identify TCP; TCP’s addresses and ports identify the connection. The SMTP process then interprets application bytes. IP alone does not decide which mailbox receives the email.

## 4 Email architecture and message structure

### Submission relay and access

A mail user agent helps a person compose and read messages. A mail submission agent accepts outgoing mail from a user or application under submission policy. A mail transfer agent relays messages between mail systems. A local delivery agent places accepted mail into a mailbox. One software product can perform several of these roles; they are functional distinctions rather than requirements for separate machines.

Message submission commonly uses port 587 with STARTTLS. Implicit TLS submission uses port 465, where TLS begins immediately after TCP establishment. Server-to-server SMTP relay normally uses port 25, possibly upgraded with STARTTLS. These uses are distinguished by RFC 6409 and RFC 8314. Authentication of a user is particularly relevant at submission; arbitrary Internet sending MTAs do not normally authenticate to each destination using Bob’s password.

Bob’s access connection is distinct from SMTP relay. IMAP keeps mailbox state on the server and supports folders and synchronization. POP3 provides a download-oriented access model. Webmail uses browser-to-service HTTP over TLS, while the service handles mailbox operations internally. Receiving SMTP on port 25 is not the same as an email application downloading mail.

### The envelope and the visible message

SMTP carries an envelope and message content. The envelope contains the reverse path supplied by MAIL FROM and recipient paths supplied by RCPT TO. The content contains message header fields, a blank line, and a body. Message headers such as From, To, Date, Subject, and Message-ID are application data. They are not IP or TCP headers.

The envelope recipient can differ from the visible To field. A blind-copy recipient can receive a message even when that mailbox is not listed in the visible message. Forwarding and mailing lists can also create differences. Delivery decisions use the envelope recipients and server policy, not just the human-readable To field. The reverse path is used for delivery-error handling and can be empty in some messages such as delivery status notifications, helping avoid error loops.

```text
SMTP envelope
  MAIL FROM:<alice@sender.example>
  RCPT TO:<bob@recipient.example>

Message content
  From: Alice <alice@sender.example>
  To: Bob <bob@recipient.example>
  Date: Wed, 18 Nov 2026 09:00:00 -0500
  Subject: Review example
  Message-ID: <review-001@sender.example>

  Hello Bob.
```

The blank line separates message header fields from the body. Internet message syntax is specified in RFC 5322. MIME extends content representation with media types, multipart boundaries, and transfer encodings. For example, an attachment can be encoded as base64 within a MIME body part. This increases the application byte count before TCP sees it. A filename or base64 string is still content; it does not determine the IP route.

Historically, basic SMTP content is restricted to a seven-bit representation. Extensions such as 8BITMIME and SMTPUTF8 change relevant capabilities when both peers support them. A realistic sender examines the advertised capabilities rather than assuming arbitrary binary files can be inserted into a basic SMTP DATA transfer unchanged.

## 5 DNS and the destination mail server

### From a recipient domain to a host address

For ordinary direct relay, the sending MTA queries the recipient domain’s MX records. The result supplies mail exchanger names and preferences. The sender then obtains A or AAAA records for a chosen exchanger to obtain an IPv4 or IPv6 address. Resolving recipient.example as an ordinary website address is not generally equivalent to finding its mail exchanger. A user’s submission application usually contacts a configured submission host instead of resolving every recipient’s MX itself.

```text
Question: recipient.example. IN MX
Answer:
recipient.example. 3600 IN MX 10 mx1.recipient.example.
recipient.example. 3600 IN MX 20 mx2.recipient.example.

Question: mx1.recipient.example. IN A
Answer:
mx1.recipient.example. 3600 IN A 203.0.113.25
```

Here preference 10 is preferred over 20; smaller values have higher preference. The number 10 is not a port, delay, or probability. A failure of the preferred exchanger can lead the MTA to try an alternative under SMTP routing rules. Equal-preference exchangers can spread traffic. If the domain has no MX records, SMTP defines an implicit-MX fallback to the domain’s address records under the applicable conditions. An explicit null MX instead indicates that the domain does not accept mail. These cases must not be confused with a temporary DNS failure.

### What the resolver does

The MTA or host’s stub resolver normally sends a request to a recursive resolver. On a cache miss, the recursive resolver can follow referrals from root servers to the relevant top-level-domain servers and then to authoritative servers. The authoritative service supplies records for the domain. A recursive request asks the resolver to obtain the final answer; iterative exchanges return answers or referrals that let the resolver continue. It is not necessary for Alice’s host to contact every level directly.

DNS caching reduces repeated work. A record’s TTL controls how long it can be cached, subject to resolver behavior; it is unrelated to an IP packet’s TTL. An MX answer can be cached independently from the exchanger’s A or AAAA answer. A cached answer can therefore make a packet capture contain SMTP without a visible preceding DNS request.

Traditional DNS commonly uses UDP with server port 53, but DNS also uses TCP, including when responses cannot be handled in the UDP exchange. A DNS request has its own transaction identifier, question fields, and response records. Encrypted resolver transports also exist, but the base example assumes traditional DNS. DNS is not part of the SMTP TCP connection. The UDP source port used for a DNS query need not equal the ephemeral TCP port used later for SMTP.

### Why DNS traffic still needs the lower layers

Suppose A’s DNS resolver lies off the local subnet and A has an empty ARP cache. A must first determine its route to that resolver and learn the next hop’s MAC address. It then sends a frame containing an IP packet containing a UDP datagram containing a DNS query. The response returns through the corresponding layers. Only after selecting B’s address does A create the SMTP TCP connection. This dependency explains why a fully cold trace can begin with ARP before any DNS or TCP exchange.

Source anchors: official application-layer lecture topics 2.3 and 2.4; DNS record format in RFC 1035; SMTP mail routing in RFC 5321 section 5. These notes explain the ordinary route while distinguishing configuration and caching variations.

## 6 SMTP dialogue and acceptance

### A stateful application exchange over a stream

The following invented plaintext relay dialogue shows the basic exchange described by RFC 5321. C means the sending SMTP client and S means the receiving SMTP server. Each actual line ends in CRLF. The labels and indentation are explanatory and are not transmitted. The example omits TLS to make the application syntax visible; a later example adds it.

```text
S: 220 mx1.recipient.example ESMTP ready
C: EHLO mail.sender.example
S: 250-mx1.recipient.example
S: 250 SIZE 10485760
C: MAIL FROM:<alice@sender.example>
S: 250 2.1.0 Sender accepted
C: RCPT TO:<bob@recipient.example>
S: 250 2.1.5 Recipient accepted
C: DATA
S: 354 End content with CRLF dot CRLF
C: From: Alice <alice@sender.example>
C: To: Bob <bob@recipient.example>
C: Date: Wed, 18 Nov 2026 09:00:00 -0500
C: Subject: Review example
C: Message-ID: <review-001@sender.example>
C:
C: Hello Bob.
C: ..This body line begins with a dot.
C: .
S: 250 2.0.0 Message accepted for delivery
C: QUIT
S: 221 2.0.0 Closing connection
```

EHLO identifies the client and asks for ESMTP capabilities. A hyphen after a reply code indicates a continuation line; the final line uses a space. The SIZE extension in this example advertises a size limit. MAIL FROM starts the envelope transaction, RCPT TO specifies a recipient, and DATA requests permission to transmit content. The body line beginning with a period is dot-stuffed: an extra leading period is transmitted and removed by the receiver. The terminator is CRLF followed by a single period followed by CRLF. The dot terminator is SMTP framing, not part of the delivered message body.

The server’s final 250 after the content is the key application acceptance point. A 250 reply to MAIL FROM alone does not mean a message has been accepted. A TCP ACK only confirms transport receipt of stream bytes at the peer; it says nothing about mailbox delivery. Even the final SMTP acceptance does not mean Bob has read the message. A relay may still have forwarding or local delivery work to do.

The first digit of an SMTP reply identifies its broad class: 2 indicates success, 3 indicates an intermediate positive response requiring further input, 4 indicates a temporary failure, and 5 indicates a permanent failure for that request. For instance, 451 can cause queuing and retry, whereas 550 for a nonexistent recipient normally requires treating that recipient as rejected. Mixed results are possible when a transaction names several recipients.

### Adding security and distinguishing the handshakes

With STARTTLS, the client first establishes TCP, reads the SMTP greeting, sends EHLO, and sees STARTTLS advertised. It sends STARTTLS, receives 220, and then runs a TLS handshake on the existing TCP connection. After successful negotiation, it sends a new EHLO over the protected channel. Earlier capability information is discarded as required by the STARTTLS specification. A submission client can then authenticate according to the service’s policy.

```text
TCP three way handshake
SMTP greeting and EHLO capability exchange
C: STARTTLS
S: 220 Ready to start TLS
TLS handshake on the same TCP connection
Encrypted SMTP: EHLO, optional AUTH, MAIL, RCPT, DATA
```

With implicit TLS on submission port 465, the TLS handshake happens before the SMTP greeting. A TLS handshake, TCP handshake, and SMTP greeting exchange are different events at different protocol boundaries. TLS changes TCP payload visibility and byte counts; IP routing and link delivery still operate underneath. A capture can show TCP ports, sequence numbers, and packet sizes while the SMTP commands are encrypted.

TLS normally protects one connection between two participating systems. If a mail relay terminates TLS and opens another connection, that is another security association. Transport encryption does not by itself provide message encryption all the way to Bob’s reading application. STARTTLS behavior is defined in RFC 3207; submission and access TLS guidance is in RFC 8314.

## 7 TCP connection reliability and control

### Establishing connection state

TCP provides a full-duplex, ordered byte stream. Before ordinary data transfer, the endpoints establish connection state using the three-way handshake. In the running example, choose client initial sequence number 1000 and server initial sequence number 7000. These are invented absolute numbers; packet analyzers often display relative numbers instead.

| Step | Direction | Flags | Sequence | ACK number | Meaning |
| --- | --- | --- | --- | --- | --- |
| 1 | A to B | SYN | 1000 | Not valid | Propose client sequence space |
| 2 | B to A | SYN ACK | 7000 | 1001 | Accept client SYN and propose server space |
| 3 | A to B | ACK | 1001 | 7001 | Acknowledge server SYN |

SYN consumes one sequence number, even without application payload. After this exchange, the next client data byte is numbered 1001 and the next server data byte is numbered 7001. Each direction has its own sequence space. The server usually sends its SMTP greeting after TCP establishment; the greeting is ordinary TCP payload and advances the server’s sequence number.

![Figure 3  TCP establishment followed by greeting and EHLO bytes](figures/sequence.png)

Figure 3  TCP establishment followed by greeting and EHLO bytes

### Sequence numbers count bytes

The TCP sequence field identifies the first payload byte in a data-bearing segment. A cumulative ACK identifies the next byte expected, acknowledging all earlier bytes in order. Suppose a client segment starts at sequence 1001 and contains 26 bytes. It occupies byte numbers 1001 through 1026, and the receiver can acknowledge 1027. An ACK number is not a segment counter and does not count only SMTP body bytes. All TCP payload bytes count, including greetings, commands, reply lines, and TLS records when present.

If a segment is lost, the receiver cannot deliver bytes beyond a gap as an uninterrupted stream to the application. It may buffer out-of-order data and repeat its cumulative ACK. The sender retransmits missing bytes using a retransmission timer or loss signals such as duplicate ACKs. Selective acknowledgment, when negotiated, can report additional received ranges. A retransmission uses the same stream byte positions, although its segment boundaries need not be identical.

A checksum detects accidental corruption in TCP header and data using a pseudoheader that also includes IP addressing information. It is not a cryptographic integrity check. Duplicates are recognized through sequence space, and data are delivered once in order to the application when TCP succeeds. TCP cannot promise success after an indefinitely broken path; it can eventually report failure.

### Flow control and congestion control

Flow control protects the receiving endpoint’s buffers. The receiver advertises a receive window, rwnd, representing how much additional data it can accept under TCP’s window rules. If the SMTP process reads slowly, available receive space can shrink. The advertised field can be interpreted with a negotiated window scaling option; an unscaled raw field alone does not always give the effective window in bytes.

Congestion control protects the network from excessive offered load. The sender maintains a congestion window, cwnd, and adjusts it based on acknowledgments and congestion evidence. The classic model includes slow start, congestion avoidance, and reductions after loss. Actual operating systems may use different algorithms. A useful simplified bound is that bytes outstanding cannot exceed min(rwnd, cwnd), and new sending must also account for data already in flight.

For example, if rwnd is 32 KiB and cwnd is 12 KiB, at most roughly 12 KiB can be outstanding under this simplified model. If 8 KiB are already outstanding, only about 4 KiB of additional new data fit. Increasing the receiver buffer does not remove the congestion-window constraint. On a stable path, a rough window-limited throughput bound is window divided by RTT, subject to link capacity and application behavior.

Kurose and Ross’s transport chapter develops reliability and control as separate mechanisms. RFC 9293 specifies TCP’s core behavior; RFC 5681 describes classic congestion-control algorithms. The numeric examples here are independently chosen to explain their implications.

### TCP header fields in context

| Field | Typical size | Interpretation |
| --- | --- | --- |
| Source and destination ports | 16 bits each | Identify transport endpoints |
| Sequence and ACK numbers | 32 bits each | Locate bytes and the next expected byte |
| Data offset | 4 bits | TCP header length in 32 bit words |
| Flags | Control bits | SYN, ACK, FIN, RST, PSH and others |
| Receive window | 16 bits | Advertised receive space before scaling |
| Checksum | 16 bits | Error detection over segment and pseudoheader |
| Urgent pointer | 16 bits | Used with urgent signaling |
| Options | Variable | MSS, scaling, SACK permission, timestamps |

The minimum TCP header is 20 bytes. MSS expresses a maximum TCP payload size the peer is willing to receive in a segment; it is not the Ethernet frame size. TCP must also respect the usable path MTU. PSH does not delimit an SMTP message and is not a guarantee that a particular application read returns one entire command.

### Closing and failure

FIN means that an endpoint has no more bytes to send in that direction. It consumes one sequence number and is acknowledged. Because TCP is full duplex, each direction can close separately; four segments are common in an orderly close, but ACK and FIN can be combined. The active closer normally remains in TIME-WAIT to handle delayed traffic and final acknowledgment recovery. RST terminates a connection abruptly under the applicable conditions.

QUIT and the SMTP 221 reply are application messages. FIN and ACK are transport control. A TCP connection can fail without a polite QUIT. If a connection disappears after content transmission but before the client receives the final SMTP reply, the sender can face uncertainty about acceptance. Retrying can produce a duplicate message. Reliable byte transfer alone does not create exactly-once application processing across crashes and reconnects.

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

## 17 Official lecture map and references

### How these notes connect to the authors materials

The author-maintained site groups the relevant topics as follows. Topic labels and numbering can vary by textbook edition. The links below point to verified official chapter pages; use the topic titles when matching your course materials. The chapter 6 page supplies notes for switched LANs, and the chapter 7 page consulted supplies knowledge-check entries for Wi-Fi. Those entries should not be described as verified full video lectures.

| Official chapter and topics | Connection to these notes |
| --- | --- |
| 1 Computer Networks and the Internet | Layering, encapsulation, delay, packet switching |
| 2.1 application principles and 2.6 socket programming | Processes, sockets, transport services |
| 2.3 Email and 2.4 DNS | SMTP infrastructure, message exchange, name resolution |
| 3.1 to 3.5 transport and TCP | Demultiplexing, reliability, sequence numbers, timers, windows |
| 3.6 and 3.7 congestion control | Network load and congestion window |
| 4 network data plane | Router forwarding, IP, addressing, NAT |
| 5 network control plane | Routing versus forwarding, ICMP context |
| 6.2 and 6.4 link layer and LANs | CRC, ARP, Ethernet, switches, VLANs |
| 6.7 retrospective | Integrating supporting protocols in one application journey |
| 7.3 WiFi and 7.7 higher layer impact | Wireless framing, shared medium, local retries |

### Author maintained lecture and video links

- [Kurose and Ross official online lecture index](https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm)
- [Chapter 1 Computer Networks and the Internet](https://gaia.cs.umass.edu/kurose_ross/videos/1/)
- [Chapter 2 The Application Layer](https://gaia.cs.umass.edu/kurose_ross/videos/2/)
- [Official Email video linked from Chapter 2](https://www.youtube.com/watch?v=D3GMrOMR2dk)
- [Official DNS video linked from Chapter 2](https://www.youtube.com/watch?v=6lRcMh5Yphg)
- [Official application principles video linked from Chapter 2](https://www.youtube.com/watch?v=abeupgK5z48)
- [Official socket programming video linked from Chapter 2](https://www.youtube.com/watch?v=_iHMMo7SDfQ)
- [Chapter 3 Transport Layer](https://gaia.cs.umass.edu/kurose_ross/videos/3/)
- [Chapter 4 Network Layer Data Plane](https://gaia.cs.umass.edu/kurose_ross/videos/4/)
- [Chapter 5 Network Layer Control Plane](https://gaia.cs.umass.edu/kurose_ross/videos/5/)
- [Chapter 6 Link Layer](https://gaia.cs.umass.edu/kurose_ross/videos/6/)
- [Chapter 7 Wireless and Mobile Networks](https://gaia.cs.umass.edu/kurose_ross/videos/7/)
- [Official textbook slide resources for edition alignment](https://gaia.cs.umass.edu/kurose_ross/ppt.php)

### Protocol specifications used for precise distinctions

- [RFC 5321 Simple Mail Transfer Protocol](https://www.rfc-editor.org/rfc/rfc5321) — dialogue, acceptance, transparency, relay routing; especially sections 3, 4, and 5.
- [RFC 5322 Internet Message Format](https://www.rfc-editor.org/rfc/rfc5322) — message headers, blank line, and body syntax.
- [RFC 2045 MIME Part One](https://www.rfc-editor.org/rfc/rfc2045) — content representation and base64 transfer encoding.
- [RFC 1035 DNS implementation and specification](https://www.rfc-editor.org/rfc/rfc1035) — resource records and message format, including MX.
- [RFC 7505 Null MX](https://www.rfc-editor.org/rfc/rfc7505) — explicit indication that a domain accepts no mail.
- [RFC 6409 Message Submission for Mail](https://www.rfc-editor.org/rfc/rfc6409) — submission versus relay and port 587.
- [RFC 3207 SMTP STARTTLS](https://www.rfc-editor.org/rfc/rfc3207) — upgrade sequence and reset of protocol knowledge.
- [RFC 8314 TLS for Email Submission and Access](https://www.rfc-editor.org/rfc/rfc8314) — implicit TLS submission and access protection.
- [RFC 9293 Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293) — TCP header, state machine, byte stream, acknowledgments, and closing.
- [RFC 5681 TCP Congestion Control](https://www.rfc-editor.org/rfc/rfc5681) — classic congestion-control behavior.
- [RFC 791 Internet Protocol](https://www.rfc-editor.org/rfc/rfc791) — IPv4 header and forwarding foundations.
- [RFC 8200 IPv6 Specification](https://www.rfc-editor.org/rfc/rfc8200) — base header and fragmentation distinctions.
- [RFC 826 Address Resolution Protocol](https://www.rfc-editor.org/rfc/rfc826) — IPv4-to-link-address resolution on a local link.
- [RFC 4861 IPv6 Neighbor Discovery](https://www.rfc-editor.org/rfc/rfc4861) — neighboring link addresses, routers, and reachability.
- [RFC 2131 Dynamic Host Configuration Protocol](https://www.rfc-editor.org/rfc/rfc2131) — initial IPv4 configuration context.

The protocol documents are foundational references, some with later updates and errata. The notes focus on the core mechanisms needed to explain the review scenario. The numeric packet trace assumes no TLS, no NAT, no options, and no fragmentation unless a section explicitly introduces a variation.
