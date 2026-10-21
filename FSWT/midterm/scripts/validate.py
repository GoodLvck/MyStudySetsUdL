"""Validate FSWT midterm exercise contracts, references, coverage, and links."""
from collections import Counter
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
topics = json.loads((root / 'topics.json').read_text())
max_pages = {'control_flow': 41, 'domain_testing': 34, 'system_integration': 38, 'system_categories': 31}
totals = Counter()
for topic in topics:
    slug, source = topic['slug'], topic['source']
    html = (root / f'{slug}.html').read_text()
    assert f'data-set="{slug}"' in html and 'lang="en"' in html
    assert 'class="tab active" data-mode="apuntes"' in html
    notes = (root / 'studysets' / f'resumen_{slug}.md').read_text()
    assert source in notes and len(notes) > 6500, slug
    coverage = []
    for suffix, counter in [('quiz', 'quiz'), ('fillblanks', 'fill')]:
        data = json.loads((root / 'studysets' / f'{slug}_{suffix}.json').read_text())
        questions = data['questions']
        assert data['metadata']['language'] == 'en'
        assert data['metadata']['sourceFiles'] == [source]
        assert data['metadata']['format'] == suffix
        assert data['metadata']['totalQuestions'] == len(questions) == topic[counter]
        assert [q['id'] for q in questions] == list(range(1, len(questions) + 1))
        covered = {q['topic'] for q in questions}
        assert covered == set(data['metadata']['topics'])
        coverage.append(covered)
        field = 'question' if suffix == 'quiz' else 'sentence'
        assert len({q[field] for q in questions}) == len(questions), (slug,suffix,'duplicate')
        for q in questions:
            assert q[field] and q['explanation']
            assert q['difficulty'] in ('easy', 'medium', 'hard')
            assert q['source']['file'] == source
            pages = [int(n) for n in re.findall(r'\d+', q['source']['pages'])]
            assert pages and all(1 <= n <= max_pages[slug] for n in pages)
            if suffix == 'quiz':
                assert len(q['options']) == len(set(q['options'])) == 4
                assert [o[:3] for o in q['options']] == ['A) ', 'B) ', 'C) ', 'D) ']
                assert q['correctAnswer'] in 'ABCD'
            else:
                assert q['sentence'].count('___') == len(q['blanks']) == len(q['hints']) == len(q['acceptedAnswers'])
                assert not re.search(r'_{4,}', q['sentence'])
                assert all(q['blanks']) and all(q['hints'])
                for answer, alternatives in zip(q['blanks'], q['acceptedAnswers']):
                    assert answer not in alternatives and len(alternatives) == len(set(alternatives))
        if suffix == 'quiz':
            distribution = Counter(q['correctAnswer'] for q in questions)
            assert len(distribution) == 4 and max(distribution.values()) - min(distribution.values()) <= 1
        totals[counter] += len(questions)
    assert coverage[0] == coverage[1], (slug, 'mode topic coverage differs', coverage)

for path in root.glob('*.html'):
    for target in re.findall(r'(?:src|href)="([^"#]+)"', path.read_text()):
        assert (path.parent / target).exists(), (path.name, target)
for path in (root / 'studysets').glob('*.md'):
    assert not re.search(r'\b(?:respondidas|preguntas|respuesta|completar frases)\b', path.read_text(), re.I)
mock = (root / 'studysets' / 'mock_exam.md').read_text()
assert [int(n) for n in re.findall(r'^\*\*(\d+)\.', mock, re.M)] == list(range(1,31))
assert totals == {'quiz':130, 'fill':84}
assert 'href="midterm/"' in (root.parent / 'index.html').read_text()
assert 'href="FSWT/midterm/"' in (root.parents[1] / 'index.html').read_text()
script = (root / 'study.js').read_text()
assert 'fswt_midterm_' in script and 'cg_' not in script
print(f'Validated {len(topics)} topic sets, {totals["quiz"]} quizzes, {totals["fill"]} blanks, 30 mock items, sources, topic coverage, counts, and local links.')
