# NCC — Net-Centric Computing

English preparation for the November 18, 2026 networking review. Follow an SMTP message from a process through the OSI and TCP/IP stacks and up to the receiving process.

## Study structure

- Six topic pages, each with **Study notes**, **Multiple choice**, and **Fill in the blanks** tabs.
- 96 original multiple-choice questions with explanations and lecture-note section references.
- 72 fill-in-the-blank exercises with hints, explanations, and accepted answer variants.
- Detailed English notes retain the full material, diagrams, packet headers, SMTP dialogue, and numerical examples.
- The complete lecture notes are readable at `notes.html`; the Word edition is available as a download.

The layout, note filenames, JSON schemas, theme controls, shuffle, navigation, mastery tracking, and session restoration follow the existing English CG and FSWT class interfaces. Browser storage uses `ncc_` keys to keep progress separate from other classes. The home page links to NCC.

Serve the repository over HTTP and open `NCC/`. The browser fetches local JSON and Markdown; opening the HTML through a file URL is insufficient in browsers that block these fetches. No build or external API is required. The Markdown parser and its license are bundled locally.

## Sources and limits

The accompanying notes follow Kurose and Ross, Computer Networking A Top-Down Approach, the official author-maintained lecture topic pages, and cited protocol specifications. The course textbook edition was not supplied. These are original explanations and practice, not official exam questions or video transcripts. Sources in exercise feedback refer to the corresponding numbered lecture-note section rather than invented textbook page numbers. Section 17 provides the official resource map.

The originally requested `Complete_Networking_Notes_SMTP_OSI_TCPIP_English.docx` was not accessible. The retained expanded Word document records that limitation; it has not been merged with that original. The topic notes and browser reader use the complete expanded material.

## Validation

Run `python3 NCC/scripts/validate.py` from the repository root to check topic coverage, exercise schemas, counts, answer variants, source sections, assets, images, and navigation. The browser checks also exercise each topic in all three modes and verify feedback, progress persistence, navigation, themes, and mobile layout.
