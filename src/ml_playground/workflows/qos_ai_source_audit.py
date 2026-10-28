"""Freeze additional implementation excerpts and measure retrieval independently of generation."""
import argparse,json,subprocess
from pathlib import Path
from qos_ai_qwen_repair import read,write,digest
from qos_ai_source_retrieval import SourceBM25
from qos_ai_retrieval import BM25

QUERIES=[
 ('sol/lib/rand.sol','Rand.next4 s','How does Rand.next4 advance deterministic random state in Sol?'),
 ('sol/lib/rand.sol','uniform = fn','Which Sol random helper produces an integer below hi?'),
 ('sol/lib/rand.sol','unit = fn','What is the range of Rand.unit in sol/lib/rand.sol?'),
 ('sol/lib/rand.sol','gauss4 = fn','How does sol/lib/rand.sol approximate a Gaussian sample?'),
 ('sol/lib/matrix.sol','mFlatten c rss','How does the Sol matrix library handle ragged input rows?'),
 ('sol/lib/matrix.sol','mGet i j m','How does reading a cell in sol/lib/matrix.sol preserve the linear matrix?'),
 ('sol/lib/matrix.sol','mDims m','What does mDims return in the Sol matrix library?'),
 ('sol/lib/matrix.sol','mSelectRows is m','Does row selection in sol/lib/matrix.sol return the source matrix too?'),
 ('sol/lib/plot.sol','P.animDots','Which Sol plotting API animates moving dots along tracks?'),
 ('sol/lib/plot.sol','P.scatter','How can I make an SVG scatter plot with sol/lib/plot.sol?'),
 ('sol/lib/plot.sol','Animation-over-time is SMIL','Does the Sol plot library use JavaScript to animate SVG charts?'),
 ('sol/lib/plot.sol','ptsBounds pts','How are degenerate point bounds handled in sol/lib/plot.sol?'),
 ('sol/lib/archive.sol','defaultLimits =','What are the default entry and byte limits in sol/lib/archive.sol?'),
 ('sol/lib/archive.sol','NativeArchive.afterCommit','When does the Sol native archive creation execute?'),
 ('sol/lib/archive.sol','createWith (Limits','How can limits be supplied to archive creation in Sol?'),
 ('sol/lib/archive.sol','create source target','Which limits does create use in sol/lib/archive.sol?'),
 ('sol/lib/git.sol','keepWorktree =','Which Sol Git uncommit mode keeps the working tree?'),
 ('sol/lib/git.sol','headLater (Repo','How can sol/lib/git.sol read the head after deferred changes?'),
 ('sol/lib/git.sol','query operation','Do queries in sol/lib/git.sol see live or deferred state?'),
 ('sol/lib/git.sol','status r =','What status format does the Sol native Git library return?'),
]


def build(a):
 out=Path(a.output)
 if out.exists():raise ValueError('refusing to replace a frozen audit')
 out.mkdir(parents=True)
 root=Path(a.source);baseline=read(a.baseline);expanded=list(baseline);files=[]
 for relative in sorted({p for p,_,_ in QUERIES}):
  p=root/relative;lines=p.read_text().splitlines();files.append({'path':relative,'sha256':digest(p)})
  for start in range(0,len(lines),30):
   end=min(start+40,len(lines));path='fprisc/'+relative
   expanded.append({'path':path,'text':f'# File: {path} lines {start+1}-{end}\n'+'\n'.join(lines[start:end]),
     'authority':'implementation','line_start':start+1,'line_end':end})
 queries=[]
 for i,(relative,needle,q) in enumerate(QUERIES):
  assert needle in (root/relative).read_text(),needle
  queries.append({'id':f'retrieval-{i:02d}','question':q,'path':'fprisc/'+relative,'evidence_substring':needle})
 write(out/'corpus.jsonl',expanded);write(out/'queries.jsonl',queries)
 results={}
 for label,idx in [('previous_source',SourceBM25(baseline)),('expanded_bm25',BM25(expanded)),('expanded_source',SourceBM25(expanded))]:
  rows=[]
  for q in queries:
   hits=idx.search(q['question'],3)
   rows.append({'id':q['id'],'top1_file':bool(hits and hits[0]['path']==q['path']),
     'top3_file':any(h['path']==q['path'] for h in hits),
     'top3_evidence':any(h['path']==q['path'] and q['evidence_substring'] in h['text'] for h in hits),
     'hits':[{'path':h['path'],'text':h['text']} for h in hits]})
  results[label]={'n':len(rows),**{k:sum(r[k] for r in rows) for k in ('top1_file','top3_file','top3_evidence')},'rows':rows}
 report={'kind':'retrieval-only development audit; no model generation or training; exact source substring is a proxy for relevant evidence',
   'source_git_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
   'files':files,'baseline_sha256':digest(a.baseline),'expanded_sha256':digest(out/'corpus.jsonl'),
   'queries_sha256':digest(out/'queries.jsonl'),'original_chunks':len(baseline),'expanded_chunks':len(expanded),'results':results}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:{x:y for x,y in v.items() if x!='rows'} for k,v in results.items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('source','baseline','output'):p.add_argument('--'+name,required=True)
 build(p.parse_args())
