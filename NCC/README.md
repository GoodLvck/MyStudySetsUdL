# NCC — Net-Centric Computing

English notes for explaining the SMTP message journey in the networking review.

## Structure

One study set follows the sending process, socket, DNS, SMTP, TCP, IP, local delivery, switches and routers, and receiving process. The OSI and TCP/IP layer mapping is included. The three familiar tabs are **Study notes**, **Multiple choice**, and **Fill in the blanks**.

There are 16 basic questions and 12 blank exercises with feedback and hints. The goal is to explain the process; sequence-number calculations, hexadecimal headers, timing calculations, and congestion algorithms are outside the main study set.

Open `NCC/` through an HTTP server. `notes.html` provides the same process notes without exercises. The existing English study interface retains shuffle, themes, local progress, and session restoration. The new set uses its own `smtp_journey` storage namespace inside NCC. Old topic URLs redirect to this set.

The earlier expanded material remains available under **Optional detailed reference** at `reference.html`, with the original expanded Word download preserved. It is not required for the main process explanation. The originally requested prior Word file was unavailable; the expanded file has not been compared with it.

Sources follow Kurose and Ross's official application, transport, network, and link topics and the sources cited in the expanded notes. These are original teaching notes and questions, not a video transcript or official exam questions.

Run `python3 NCC/scripts/validate.py` to check the study-set contracts, sources, counts, local navigation, and retained reference assets.
