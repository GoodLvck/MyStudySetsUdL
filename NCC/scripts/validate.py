"""Validate NCC content, references, assets, and the shared study-app contracts."""
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import re
import zipfile

root = Path(__file__).resolve().parents[1]
topics = json.loads((root / 'topics.json').read_text())
assert len(topics) == 6
assert len({t['slug'] for t in topics}) == len(topics)
counts = Counter()
covered = []
norm = lambda s: re.sub(r'\s+', ' ', s.lower().strip())
full = (root / 'studysets/complete_notes.md').read_text()
original = next(root.glob('*Expanded.md')).read_text()
sections = {}
matches = list(re.finditer(r'^## (\d+) (.+)$', original, re.M))
for i, m in enumerate(matches):
    end = matches[i + 1].start() if i + 1 < len(matches) else len(original)
    sections[int(m[1])] = original[m.start():end].strip()
assert list(sections) == list(range(1, 18))
assert all(s in full for s in sections.values())

for topic in topics:
    slug = topic['slug']
    page = (root / f'{slug}.html').read_text()
    assert f'data-set="{slug}"' in page and 'lang="en"' in page
    assert 'class="tab active" data-mode="apuntes"' in page
    assert all(f'data-mode="{mode}"' in page for mode in ['apuntes', 'quiz', 'fill'])
    notes = (root / 'studysets' / f'resumen_{slug}.md').read_text()
    assert len(notes.split()) > 900, slug
    assert all(sections[n] in notes for n in topic['sections'])
    covered += topic['sections']
    for suffix, key in [('quiz', 'quiz'), ('fillblanks', 'fill')]:
        data = json.loads((root / 'studysets' / f'{slug}_{suffix}.json').read_text())
        meta = data['metadata']
        items = data['questions']
        assert meta['language'] == 'en' and meta['format'] == suffix
        assert meta['sourceFiles'] == topic['sourceFiles']
        assert meta['sourceSections'] == topic['sections']
        assert len(items) == meta['totalQuestions'] == topic[key]
        assert [q['id'] for q in items] == list(range(1, len(items) + 1))
        text_key = 'question' if suffix == 'quiz' else 'sentence'
        assert len({q[text_key] for q in items}) == len(items)
        for q in items:
            assert q['topic'] == topic['title'] and len(q['explanation']) > 30
            assert q['difficulty'] in ['easy', 'medium', 'hard']
            assert q['source']['file'] in topic['sourceFiles']
            assert (root / q['source']['file']).is_file()
            section = int(re.match(r'Section (\d+) — ', q['source']['section'])[1])
            assert section in topic['sections']
            assert q['source']['section'].split(' — ', 1)[1] in sections[section]
            if suffix == 'quiz':
                assert len(q['options']) == 4
                assert [o[:3] for o in q['options']] == ['A) ', 'B) ', 'C) ', 'D) ']
                assert len({norm(o[3:]) for o in q['options']}) == 4
                assert q['correctAnswer'] in ['A', 'B', 'C', 'D']
            else:
                assert q['sentence'].count('___') == len(q['blanks']) == len(q['hints']) == len(q['acceptedAnswers'])
                assert not re.search(r'_{4,}', q['sentence'])
                assert all(q['blanks']) and all(q['hints'])
                for answer, alts in zip(q['blanks'], q['acceptedAnswers']):
                    values = [norm(a) for a in [answer] + alts]
                    assert len(values) == len(set(values))
        if suffix == 'quiz':
            balance = Counter(q['correctAnswer'] for q in items)
            assert set(balance) == set('ABCD')
            assert max(balance.values()) - min(balance.values()) <= 1
        counts[key] += len(items)

assert sorted(covered) == list(range(1, 17))
assert counts == {'quiz': 96, 'fill': 72}
assert 'ncc_' in (root / 'study.js').read_text()
assert 'cg_' not in (root / 'study.js').read_text()
assert 'ncc_theme' in (root / 'index.html').read_text()
assert 'href="NCC/"' in (root.parent / 'index.html').read_text()
for page in root.glob('*.html'):
    for target in re.findall(r'(?:href|src)="([^"#]+)"', page.read_text()):
        parsed = urlsplit(target)
        if not parsed.scheme:
            assert (page.parent / unquote(parsed.path)).exists(), (page, target)
for notes in (root / 'studysets').glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', notes.read_text()):
        parsed = urlsplit(target)
        if not parsed.scheme and parsed.path:
            # Markdown is inserted into topic pages, so URLs resolve from NCC/.
            assert (root / unquote(parsed.path)).exists(), (notes, target)
    assert '[[' not in notes.read_text()
with zipfile.ZipFile(next(root.glob('*Expanded.docx'))) as z:
    assert z.testzip() is None
assert len(list((root / 'figures').glob('*.png'))) == 4
print('Validated 6 topics, 96 quiz questions, 72 blanks, complete section coverage, source references, Word archive, images, and local links.')
