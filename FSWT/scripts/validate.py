"""Validate FSWT content contracts, references, coverage and local page assets."""
from pathlib import Path
from collections import Counter
import json,re
root=Path(__file__).resolve().parents[1]
topics=json.loads((root/'topics.json').read_text())
max_pages={'fundamentals':34,'process_models':34,'software_testing':51,'cases_scenarios':18,'unit_testing':33}
totals=Counter()
for topic in topics:
 slug=topic['slug']; source=topic['source']
 html=(root/f'{slug}.html').read_text()
 assert f'data-set="{slug}"' in html
 assert 'dragdrop' not in html
 for asset in re.findall(r'(?:src|href)="([^"#]+)"',html):
  assert (root/asset).exists(),(slug,asset)
 for suffix,key,counter in [('quiz','questions','quiz'),('fillblanks','questions','fill'),('flashcards','cards','cards')]:
  data=json.loads((root/'studysets'/f'{slug}_{suffix}.json').read_text())
  items=data[key]; count_key='totalCards' if key=='cards' else 'totalQuestions'
  assert len(items)==data['metadata'][count_key]==topic[counter]
  assert [x['id'] for x in items]==list(range(1,len(items)+1))
  assert set(x['topic'] for x in items)==set(data['metadata']['topics'])
  assert data['metadata']['sourceFiles']==[source]
  field={'quiz':'question','fillblanks':'sentence','flashcards':'front'}[suffix]
  assert len({x[field] for x in items})==len(items),('duplicate',slug,suffix)
  for x in items:
   assert x['topic'] and x[field]
   assert x['source']['file']==source
   assert all(0<int(n)<=max_pages[slug] for n in re.findall(r'\d+',x['source']['pages']))
   if suffix=='quiz':
    assert x['difficulty'] in ['easy','medium','hard']
    assert len(x['options'])==4 and len(set(x['options']))==4
    assert [o[:3] for o in x['options']]==['A) ','B) ','C) ','D) ']
    assert x['correctAnswer'] in 'ABCD' and x['explanation']
   elif suffix=='fillblanks':
    assert x['sentence'].count('___')==len(x['blanks'])==len(x['hints'])
    assert all(x['blanks']) and all(x['hints']) and x['explanation']
    if 'acceptedAnswers' in x: assert len(x['acceptedAnswers'])==len(x['blanks'])
   else: assert x['back']
  if suffix=='quiz':
   distribution=Counter(x['correctAnswer'] for x in items)
   assert max(distribution.values())-min(distribution.values())<=1
  totals[counter]+=len(items)
 notes=(root/'studysets'/f'resumen_{slug}.md').read_text()
 assert Path(source).name in notes
 assert len(notes)>4000
 # All exercise topics should be available in the other retrieval modes.
 quiz=json.loads((root/'studysets'/f'{slug}_quiz.json').read_text())
 cards=json.loads((root/'studysets'/f'{slug}_flashcards.json').read_text())
 fill=json.loads((root/'studysets'/f'{slug}_fillblanks.json').read_text())
 assert set(quiz['metadata']['topics'])==set(cards['metadata']['topics'])==set(fill['metadata']['topics'])
print(f'Validated {len(topics)} sets: {dict(totals)}; JSON contracts, topic coverage, page references, counts, answer balance, notes and assets OK.')
