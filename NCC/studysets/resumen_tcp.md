# TCP Reliability and Control

Full explanations from lecture-note sections 7. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

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

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Transport layer](https://gaia.cs.umass.edu/kurose_ross/videos/3/)
- [TCP specification](https://www.rfc-editor.org/rfc/rfc9293)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
