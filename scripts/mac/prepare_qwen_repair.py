"""Freeze prompts/curriculum before model answers. No evaluation labels in prompt files."""
import argparse,json,random,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from qos_ai_source_retrieval import SourceBM25,grounded_question
from qos_ai_retrieval import BM25,with_reference
from qos_ai_qwen_repair import read,write,digest,score_row

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--dataset',default='dataset');p.add_argument('--output',default='prepared');a=p.parse_args()
r=Path(a.root);out=r/a.output
if out.exists():raise ValueError('prepared inputs already exist')
out.mkdir()
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
corpus=read(r/'source_corpus.jsonl');idx=SourceBM25(corpus);lex=BM25(corpus)
legacy=BM25(read(r/'legacy_corpus.jsonl'))
docs=read(r/'documentation_rows.jsonl')
for row in docs:
 if not score_row(row,row['answer'],'unused')['rubric_proxy_pass']:raise ValueError(('reference does not satisfy rubric',row['id']))
def prompt(row,index=None,ground=False,budget=900):
 q=row['question'];paths=[]
 if row['kind']=='docs' and index:
  hits=index.search(q,3)
  ref=tok.decode(tok.encode('\n\n'.join(h['text'] for h in hits),add_special_tokens=False)[:budget])
  q=with_reference(grounded_question(q) if ground else q,ref);paths=[h['path'] for h in hits]
 return {'id':row['id'],'suite':row['suite'] if 'suite' in row else 'fresh','question':row['question'],'prompt':q,'references':paths,'max_new_tokens':row.get('max_new_tokens',320)}
for split in ('train','valid','dev','final'):
 data=read(r/a.dataset/f'{split}.jsonl')+[d for d in docs if d['split']==split]
 write(out/f'{split}_rows.jsonl',data)
 if split in ('train','valid'):
  data=[{**row,**prompt(row,idx,True,550)} for row in data]
  random.Random(2026102702).shuffle(data)
  write(out/f'{split}_prepared.jsonl',data)
 else:
  write(out/f'{split}_source_prompts.jsonl',[prompt(row,idx,True) for row in data])
  dd=[row for row in data if row['kind']=='docs']
  for label,index,ground in [('closed',None,False),('legacy',legacy,False),('lexical',lex,False)]:
   write(out/f'{split}_docs_{label}_prompts.jsonl',[prompt(row,index,ground) for row in dd])
manifest={'policy':'frozen before model generation; final split used once after development selection','files':{p.name:digest(p) for p in out.glob('*.jsonl')},'corpus_sha256':digest(r/'source_corpus.jsonl'),'legacy_corpus_sha256':digest(r/'legacy_corpus.jsonl')}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
