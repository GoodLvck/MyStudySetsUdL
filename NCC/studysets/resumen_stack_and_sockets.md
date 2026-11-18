# Stacks Processes and Sockets

Full explanations from lecture-note sections 1, 2, 3. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

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

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Application layer](https://gaia.cs.umass.edu/kurose_ross/videos/2/)
- [Introduction and layering](https://gaia.cs.umass.edu/kurose_ross/videos/1/)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
