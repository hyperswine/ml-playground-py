import json
from pathlib import Path
from qos_ai_data import digest
from qos_ai_bench import reviewed_scores

benchmark={r['id']:r for r in map(json.loads,Path('qos_ai_benchmark/benchmark.jsonl').read_text().splitlines())}
old_reviews={r['id']:r for r in map(json.loads,Path('reports/dev_smoke_greedy/source_review.jsonl').read_text().splitlines())}
ids=set(old_reviews)
explanations={
 'fact-ownership-2':(False,'Does not distinguish repository ownership and incorrectly substitutes QOS Portable running the compiler.'),
 'fact-profiles-3':(False,'Calls the default profile FP-RISC instead of Base.'),
 'fact-qos-pin-3':(True,'Correct core fact: the commit is recorded in fprisc.lock.json for releases. Omits the clean/compiler qualifiers, so the stricter lexical rubric fails.'),
 'fact-qos-platform-3':(False,'Incorrectly says QOS Portable is not designed for Linux; source supports Linux and FreeBSD/arm64.'),
 'fact-call-wire-1':(False,'Generic send/reply description does not answer the request/reply wire shape or correlation-id mechanism.'),
 'fact-binary-bounds-3':(False,'Invents Result Int String and out-of-bytes errors instead of None.'),
 'fact-option-result-1':(False,'Does not explain ordinary absence versus diagnostic failure; substitutes an unsupported Option(Int)/Result statement.'),
 'fact-json-trailing-1':(True,'Correctly answers No to accepting a JSON value followed by extra text, with a source-supported generic parse-error explanation. Does not explicitly describe trailing-character rejection.'),
 'output-string-slice-1':(False,'Predicts 1 instead of bcd, and emits malformed irrelevant code.'),
 'debug-at':(False,'Does not correct the zero-based index or identify the first-byte value 65.'),
 'write-clamp':(False,'Invents clamping syntax, does not use Numeric.clamp and predicts 20 instead of 8. Generated code was not executed.'),
 'unknown-fpr-root':(False,'Does not acknowledge missing flags/environment/layout or explain how the absolute checkout is resolved.')}
summary={'selection':'The same 12 category/concept sample IDs used before fine-tuning; fixed independently of results.',
 'reviewer':'Codex source review, not independent human review',
 'criterion':'Answers the direct question correctly with no material false claim; minor omitted qualifiers may be acceptable and are noted. Executable correctness is not inferred.',
 'runs':{}}
for arm in ('parent_raw','tuned_raw'):
 report=Path('reports/sft5_'+arm)
 rows=[json.loads(x) for x in (report/'answers.jsonl').read_text().splitlines()]
 judgments=[]
 for row in rows:
  if row['id'] not in ids:continue
  ref=benchmark[row['id']]
  source=ref['sources'][0]
  if arm=='parent_raw':
   correct=False
   explanation=old_reviews[row['id']]['evidence'].split(' Reference: ')[0]
  else:
   correct,explanation=explanations[row['id']]
  judgments.append({'id':row['id'],'answer_hash':digest(row['answer']),
   'reviewer':summary['reviewer'],'factual_correct':correct,'generated_code_compiles':None,
   'generated_code_behavior_correct':None,
   'evidence':explanation+' Reference: '+source['path']+':'+str(source['line_start'])+'. '+ref['answer']})
 assert len(judgments)==12
 path=report/'source_review.jsonl'
 path.write_text(''.join(json.dumps(r)+'\n' for r in judgments))
 scores=reviewed_scores(rows,judgments)
 artifact={'answers_sha256':__import__('hashlib').sha256((report/'answers.jsonl').read_bytes()).hexdigest(),
  'judgments_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'metrics':scores}
 (report/'reviewed_scores.json').write_text(json.dumps(artifact,indent=2)+'\n')
 summary['runs'][arm]=scores
Path('reports/verified_sft5_source_review.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
