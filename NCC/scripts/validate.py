"""Validate the process-focused NCC study set and retained reference assets."""
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json, re, zipfile

root = Path(__file__).resolve().parents[1]
topics = json.loads((root / 'topics.json').read_text())
assert len(topics) == 1
topic = topics[0]
assert topic['slug'] == 'smtp_journey'
page = (root / 'smtp_journey.html').read_text()
assert 'data-set="smtp_journey"' in page and 'lang="en"' in page
assert all(f'data-mode="{mode}"' in page for mode in ['apuntes', 'quiz', 'fill'])
assert 'class="tab active" data-mode="apuntes"' in page
notes = (root / 'studysets/resumen_smtp_journey.md').read_text()
assert 600 < len(notes.split()) < 1500
assert re.findall(r'^## (\d+) ', notes, re.M) == [str(n) for n in range(1, 9)]
for term in ['socket', 'DNS', 'MX', 'SMTP', 'TCP', 'IP', 'ARP', 'Neighbor Discovery', 'switch', 'router', 'encapsulation', 'decapsulation', 'OSI']:
    assert term in notes, term
norm = lambda s: re.sub(r'\s+', ' ', s.lower().strip())
for suffix, count_key in [('quiz', 'quiz'), ('fillblanks', 'fill')]:
    data = json.loads((root / 'studysets' / f'smtp_journey_{suffix}.json').read_text())
    qs = data['questions']; meta = data['metadata']
    assert meta['language'] == 'en' and meta['format'] == suffix
    assert meta['sourceFiles'] == topic['sourceFiles']
    assert len(qs) == meta['totalQuestions'] == topic[count_key]
    assert [q['id'] for q in qs] == list(range(1, len(qs) + 1))
    key = 'question' if suffix == 'quiz' else 'sentence'
    assert len({q[key] for q in qs}) == len(qs)
    for q in qs:
        assert q['topic'] == topic['title'] and q['difficulty'] == 'easy'
        assert len(q['explanation']) > 30
        assert q['source']['file'] in topic['sourceFiles']
        assert (root / q['source']['file']).is_file()
        section = int(q['source']['section'].split()[-1])
        assert 1 <= section <= 8
        if suffix == 'quiz':
            assert [o[:3] for o in q['options']] == ['A) ', 'B) ', 'C) ', 'D) ']
            assert len({norm(o[3:]) for o in q['options']}) == 4
            assert q['correctAnswer'] in 'ABCD'
        else:
            assert q['sentence'].count('___') == len(q['blanks']) == len(q['hints']) == len(q['acceptedAnswers'])
            assert all(q['blanks']) and all(q['hints'])
            for answer, alts in zip(q['blanks'], q['acceptedAnswers']):
                values = [norm(a) for a in [answer] + alts]
                assert len(values) == len(set(values))
    if suffix == 'quiz':
        assert set(Counter(q['correctAnswer'] for q in qs).values()) == {4}
assert topic['quiz'] == 16 and topic['fill'] == 12
assert len(list((root / 'studysets').glob('*_quiz.json'))) == 1
assert len(list((root / 'studysets').glob('*_fillblanks.json'))) == 1
assert 'ncc_' in (root / 'study.js').read_text()
assert 'data-document="resumen_smtp_journey"' in (root / 'notes.html').read_text()
assert 'data-document="complete_notes"' in (root / 'reference.html').read_text()
assert 'Optional detailed reference' in (root / 'index.html').read_text()
assert 'href="NCC/"' in (root.parent / 'index.html').read_text()
for path in root.glob('*.html'):
    for target in re.findall(r'(?:href|src)="([^"#]+)"', path.read_text()):
        url = urlsplit(target)
        if not url.scheme:
            assert (path.parent / unquote(url.path)).exists(), (path.name, target)
for path in (root / 'studysets').glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        url = urlsplit(target)
        if not url.scheme and url.path:
            assert (root / unquote(url.path)).exists(), (path.name, target)
    assert '[[' not in path.read_text()
for slug in ['stack_and_sockets', 'smtp_and_dns', 'tcp', 'ip_and_neighbors', 'link_and_forwarding', 'end_to_end']:
    assert 'http-equiv="refresh"' in (root / f'{slug}.html').read_text()
    assert 'url=smtp_journey.html' in (root / f'{slug}.html').read_text()
with zipfile.ZipFile(next(root.glob('*Expanded.docx'))) as z:
    assert z.testzip() is None
assert len(list((root / 'figures').glob('*.png'))) == 4
print('Validated one process-focused set, 16 basic questions, 12 blanks, all eight note sections, redirects, navigation, and optional reference assets.')
