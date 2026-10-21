"""Validate the CG study pack's data, references, counts and local page assets."""
from collections import Counter
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
topics = json.loads((root / 'topics.json').read_text())
page_counts = [4, 14, 48, 76, 65, 51, 18, 17, 38]
totals = Counter()
source_files = set()
for topic in topics:
    slug = topic['slug']
    page = (root / f'{slug}.html').read_text()
    assert f'data-set="{slug}"' in page
    assert 'lang="en"' in page
    assert 'data-mode="flashcard"' not in page
    assert 'class="tab active" data-mode="apuntes"' in page
    notes = (root / 'studysets' / f'resumen_{slug}.md').read_text()
    assert len(notes) > 3500, slug
    for source in topic['sourceFiles']:
        assert source in notes
        source_files.add(source)
    for suffix, count_key in [('quiz', 'quiz'), ('fillblanks', 'fill')]:
        data = json.loads((root / 'studysets' / f'{slug}_{suffix}.json').read_text())
        items = data['questions']
        assert data['metadata']['language'] == 'en'
        assert data['metadata']['sourceFiles'] == topic['sourceFiles']
        assert data['metadata']['format'] == suffix
        assert len(items) == data['metadata']['totalQuestions'] == topic[count_key]
        assert [q['id'] for q in items] == list(range(1, len(items) + 1))
        field = 'question' if suffix == 'quiz' else 'sentence'
        assert len({q[field] for q in items}) == len(items)
        for q in items:
            assert q['topic'] == topic['title'] and q['explanation']
            assert q['difficulty'] in ('easy', 'medium', 'hard')
            source = q['source']['file']
            assert source in topic['sourceFiles']
            pages = [int(n) for n in re.findall(r'\d+', q['source']['pages'])]
            assert pages and all(1 <= n <= page_counts[int(source[:2])] for n in pages), (slug,q['id'],pages)
            if suffix == 'quiz':
                assert len(q['options']) == len(set(q['options'])) == 4
                assert [o[:3] for o in q['options']] == ['A) ', 'B) ', 'C) ', 'D) ']
                assert q['correctAnswer'] in 'ABCD'
            else:
                assert q['sentence'].count('___') == len(q['blanks']) == len(q['hints']) == len(q['acceptedAnswers'])
                assert not re.search(r'_{4,}', q['sentence'])
                assert all(q['blanks']) and all(q['hints'])
                for canonical, alternatives in zip(q['blanks'], q['acceptedAnswers']):
                    assert len(alternatives) == len(set(alternatives)) and canonical not in alternatives
        if suffix == 'quiz':
            balance = Counter(q['correctAnswer'] for q in items)
            assert len(balance) == 4 and max(balance.values()) - min(balance.values()) <= 1
        totals[count_key] += len(items)

for path in root.glob('*.html'):
    for target in re.findall(r'(?:src|href)="([^"#]+)"', path.read_text()):
        if not target.startswith(('https:', 'http:')):
            assert (path.parent / target).exists(), (path.name, target)
for path in (root / 'studysets').glob('*.md'):
    assert not re.search(r'\b(?:respondidas|preguntas|respuesta|completar frases)\b', path.read_text(), re.I)
mock = (root / 'studysets' / 'mock_exam.md').read_text()
assert [int(n) for n in re.findall(r'^\*\*(\d+)\.', mock, re.M)] == list(range(1,31))
assert len(source_files) == 9
assert totals == {'quiz':132, 'fill':88}
assert 'href="CG/"' in (root.parent / 'index.html').read_text()
print(f'Validated {len(topics)} sets, {totals["quiz"]} quizzes, {totals["fill"]} blanks, 30 mock questions, 9 sources, local assets and page references.')
