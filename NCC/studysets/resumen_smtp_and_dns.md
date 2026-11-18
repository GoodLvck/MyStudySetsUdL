# Email SMTP and DNS

Full explanations from lecture-note sections 4, 5, 6. The examples are original teaching scenarios; they are not captured production traffic.

[Complete lecture notes](notes.html) · [Download the Word edition](Complete_Networking_Notes_SMTP_OSI_TCPIP_English_Expanded.docx)

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

## Source connections

These notes follow Kurose and Ross, Computer Networking A Top-Down Approach. The textbook edition was not supplied; chapter topics are used instead of page numbers. The official topic pages and protocol specifications anchor the explanations. Videos are linked for reference, not transcribed.

- [Email DNS and application principles](https://gaia.cs.umass.edu/kurose_ross/videos/2/)
- [SMTP specification](https://www.rfc-editor.org/rfc/rfc5321)

See section 17 of the [complete lecture notes](notes.html) for all protocol references.
