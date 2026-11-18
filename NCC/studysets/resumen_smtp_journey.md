# The SMTP Message Journey

The review asks you to explain how a message leaves one process, travels down the network stack, crosses a network, and travels up the stack to another process. Follow the story below and explain what each layer contributes. The same pattern carries both SMTP commands and the server's replies.

## 1 Start with the sending process

Alice writes an email to Bob. A mail application submits it to a mail server, which can pass it to Bob's mail server. For the stack walkthrough, focus on the sending mail server's **SMTP process** communicating with the receiving mail server's SMTP process.

A **socket** is the interface a process uses to send and receive data through the operating system. The application gives bytes to the socket; it does not personally create electrical signals or forward packets through routers.

If Alice uses webmail, her browser usually talks to the provider using HTTPS. The provider's mail system then handles SMTP. The SMTP-sending process is the starting point of our example.

## 2 Find the receiving mail server

The sending mail server uses **DNS** to find the recipient domain's mail exchanger. An **MX record** gives a mail server's name, and an address lookup gives that server's IP address. A user's mail app may instead begin by contacting its already configured submission server.

DNS answers “Which mail server should I contact?” It does not carry the email itself. Cached information can mean no new DNS exchange is needed.

## 3 Establish TCP and begin SMTP

The sender opens a **TCP connection** to the receiving SMTP server. Server-to-server SMTP normally uses port 25. The destination port helps the receiving host identify the service, while the client uses a source port for its endpoint.

TCP establishes the connection with **SYN → SYN ACK → ACK**. It then provides an ordered, reliable byte stream: it uses acknowledgments and retransmissions to recover missing data. You only need to understand its job in the journey, rather than calculate sequence numbers.

**SMTP** supplies the email conversation. The server greets the sender, the sender identifies itself with EHLO, specifies the envelope sender with MAIL FROM and the recipient with RCPT TO, then sends the message after DATA. The server's final successful reply after the content means it has accepted responsibility for the message. The visible From, To, and Subject fields belong to the email content; they are not IP or TCP headers.

## 4 Travel down the sending stack

The SMTP process gives its bytes to TCP through the socket. **TCP adds a transport header**, including ports, and forms a segment. **IP adds a network header**, including source and destination IP addresses, and forms a packet. **Ethernet or Wi-Fi adds local link information** to carry that packet in a frame. The physical layer sends it as electrical, optical, or radio signals.

This wrapping is **encapsulation**. The basic nesting is:

```text
Link frame
  IP packet
    TCP segment
      SMTP bytes
```

An email can occupy many packets. One SMTP command does not have to fit exactly into one TCP segment. The receiving application interprets the reconstructed stream.

## 5 Cross the local network and routers

An IP address identifies the network destination. A MAC address is used for local link delivery. If the destination is on another network, the sender's first frame normally targets its **default gateway**, while the IP packet still targets the remote mail server.

For IPv4, **ARP** can discover the local next hop's MAC address. IPv6 uses **Neighbor Discovery** instead. The sender resolves a local neighbor, not the remote server's MAC address across the Internet.

A **switch** forwards frames within a LAN using MAC addresses. A **router** forwards packets between networks using destination IP addresses and its forwarding table. At a router, the incoming link frame ends and a new frame carries the packet over the next link. In the simple example without NAT, the endpoint IP addresses stay the same while the local MAC addresses change at router boundaries. The router updates hop information; it does not open the email as an SMTP recipient.

## 6 Travel up the receiving stack

At the receiving host, the physical layer recovers signals into bits. Link processing checks and removes the frame information and passes the IP packet upward. IP recognizes a local destination and passes its TCP payload to TCP. TCP checks, orders, and reconstructs the stream, then makes the bytes available to the receiving SMTP process through its socket.

This unwrapping is **decapsulation**. The receiving SMTP process interprets the commands and message content and sends replies back through the same kinds of layers in the reverse direction. The mail system can then store the accepted message in Bob's mailbox. If another mail relay is needed, it performs a new SMTP transfer over another TCP connection.

A TCP acknowledgment confirms transport receipt of bytes. It does not mean that SMTP accepted the message or that Bob read it. Bob later accesses the mailbox using a service such as IMAP or webmail; that is separate from SMTP transfer.

## 7 Name the layers in both models

The OSI model has seven layers. Kurose and Ross commonly explain the Internet with five; a four-layer TCP/IP model groups link and physical functions together.

| OSI layers | Internet model | Role in this journey |
| --- | --- | --- |
| Application, presentation, session | Application | SMTP dialogue and application representation or session functions |
| Transport | Transport | TCP and process-to-process delivery |
| Network | Network or Internet | IP addressing and routing between networks |
| Data link | Link | Ethernet or Wi-Fi frames and local delivery |
| Physical | Physical, or grouped with link in the four-layer model | Signals on the medium |

You do not need to invent separate presentation and session headers. Their functions can be handled within the Internet application implementation.

## 8 Explain the whole process aloud

“The sending SMTP process finds the receiving mail server through DNS and opens a TCP connection using a socket. SMTP handles the email conversation. TCP carries the bytes reliably, IP addresses and routes the packets, and Ethernet or Wi-Fi carries each packet over a local link using signals. Switches forward local frames, and routers move packets between networks with a new frame on each routed link. At the destination, the stack removes the encapsulation, TCP reconstructs the byte stream, and the receiving SMTP process accepts and handles the message. Its replies travel back through the stack, and the mail system delivers the message to the mailbox.”

## Sources

This explanation follows the previously prepared notes and Kurose and Ross's official topics on applications, transport, IP, and links. These are original notes, not a video transcript or official exam questions.

- [Official lectures and topic pages](https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm)
- [Application layer including email and DNS](https://gaia.cs.umass.edu/kurose_ross/videos/2/)
- [Transport layer](https://gaia.cs.umass.edu/kurose_ross/videos/3/)
- [Link layer and LANs](https://gaia.cs.umass.edu/kurose_ross/videos/6/)
