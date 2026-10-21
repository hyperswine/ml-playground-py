"""Build and verify a broader Sol/domain curriculum without moving frozen holdouts."""
import argparse
import copy
import json
import os
import platform
import random
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from qos_ai_data import (digest, file_hash, question_id, read_rows, row_split, rows_fingerprint,
                         similar_pairs, source_check, validate_manifest)


def write_rows(path, rows):
  path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def validate_extension(base, expanded):
  validate_manifest(base)
  validate_manifest(expanded)
  if expanded.get('parent_hash') != base['hash']:
    raise ValueError('Curriculum does not extend the declared frozen split')
  if expanded['files'] != base['files'] or any(expanded['questions'].get(k) != v for k,v in base['questions'].items()):
    raise ValueError('Existing file or question assignments moved')
  for name in ('corpus','qa','benchmark'):
    if expanded['inputs'][name] != base['inputs'][name]:
      raise ValueError('Benchmark or historical data changed')
  if expanded['excluded_chunks'] != base['excluded_chunks']:
    raise ValueError('Source exclusions changed')
  if any(v != 'train' for k,v in expanded['questions'].items() if k not in base['questions']):
    raise ValueError('New question assignment is not training')


def screen_extension(base, old_verified, candidates, qa, benchmark, protected_questions=()):
  """Reject new duplicates of protected material; never reassign old examples."""
  validate_manifest(base, qa=qa, verified=old_verified, benchmark=benchmark)
  protected = [r['question'] for r in qa if row_split(r, base) != 'train'] + [b['question'] for b in benchmark]
  protected += list(protected_questions)
  texts = protected + [r['question'] for r in candidates]
  parent = list(range(len(texts)))
  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i
  for a,b in similar_pairs(texts):
    a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
  blocked = {find(i) for i in range(len(protected))}
  exact_old = {question_id(r['question']):r for r in old_verified}
  old_ids = {r['id'] for r in old_verified}
  seen_questions, seen_ids = set(exact_old), set(old_ids)
  kept, rejected = list(old_verified), []
  benchmark_code = []
  for b in benchmark:
    benchmark_code += re.findall(r'```\w*\s*\n(.*?)```',b['question'],re.S)
    if b.get('program'):benchmark_code.append(b['program']['code'])
  compact=lambda text:re.sub(r'\s+','',text)
  for i,row in enumerate(candidates):
    qid=question_id(row['question'])
    reason=None
    if base['files'].get(row['path']) != 'train':reason='source file is not training'
    elif qid in base['questions'] and base['questions'][qid] != 'train':reason='protected original question'
    elif find(len(protected)+i) in blocked:reason='lexical overlap with protected question component'
    elif row.get('expression') and any(compact(row['expression']) in compact(code) for code in benchmark_code):reason='benchmark program input overlap'
    elif qid in seen_questions or row['id'] in seen_ids:reason='duplicate training question/id'
    if reason:
      rejected.append({'id':row['id'],'question':row['question'],'reason':reason})
      continue
    seen_questions.add(qid);seen_ids.add(row['id']);kept.append(row)
  expanded=copy.deepcopy(base)
  expanded.pop('hash')
  expanded['parent_hash']=base['hash']
  expanded['extension_policy']='append verified training only; preserve existing file/question/exclusion assignments'
  expanded['inputs']['verified']=rows_fingerprint(kept)
  for row in kept:
    expanded['questions'][question_id(row['question'])]='train'
  expanded['counts']['verified']={'train':len(kept)}
  expanded['extension_counts']={'original':len(old_verified),'candidates':len(candidates),'new_kept':len(kept)-len(old_verified),
                                'rejected':len(rejected),'categories':dict(Counter(r.get('category','original') for r in kept))}
  expanded['hash']=digest(expanded)
  validate_extension(base,expanded)
  return kept,rejected,expanded


def fact_candidates(old):
  # Independently written questions about training-source contracts. The benchmark
  # is consulted only by the overlap screen, not used to produce target answers.
  variants={
 'ownership':[
  'Where should I look for the language runtime when browsing the sibling projects?',
  'I need to edit a QOS device service. Which checkout owns that code?',
  'Explain the responsibility boundary between the language implementation and OS tree.',
  'Are the FP-RISC language libraries maintained with the compiler or with the OS?',
  'Someone put a compiler patch in the operating-system tree. Explain the repository ownership mistake.'],
 'profiles':[
  'For an ordinary source file with no profile header, name the implicit profile.',
  'Show the declaration that explicitly selects the normal default profile.',
  'Does an omitted profile header implicitly select Builtin? Correct this assumption.',
  'State the default profile and give its explicit spelling in one sentence.',
  'An example starts with profile base. Explain what changes if that default header is omitted.'],
 'sol-launch':[
  'Give the driver invocation for interpreting a file named demo.sol.',
  'I have an FP-RISC compiler executable and a hosted script; which subcommand runs the script?',
  'Explain how to launch a hosted Sol program from a terminal.',
  'Show the shell command for executing scripts/example.sol with the fpr driver.',
  'Which profile is selected by the fpr sol subcommand?'],
 'host-build':[
  'Describe the output artifact when a Base application is built for macOS.',
  'Does a normal Base host build require a QOS application loader?',
  'Name the runtime and machine adapter used by ordinary host executables.',
  'Explain why a Base host executable can start through the host shell.',
  'A developer expects a .qa package from a Base host build. Correct the expected artifact.'],
 'import-order':[
  'An importer-local module and FPR_HOME both contain a name. Which takes priority?',
  'Explain the role of extra FPR_PATH roots after the primary module home.',
  'Can an extra module search root override a matching importer-relative file?',
  'List the module lookup stages beginning at the importing file.',
  'An import unexpectedly uses a nearby module instead of the home copy. Explain the lookup rule.'],
 'qos-locator':[
  'What takes precedence when --fprisc and FPRISC_ROOT both name checkouts?',
  'Explain the fallback used if no explicit compiler directory or environment setting is supplied.',
  'What is fprisc.path used for in the OS checkout?',
  'List the possible configuration sources for the language checkout used by the OS driver.',
  'A sibling checkout is ignored because an environment variable names another tree. Explain why.'],
 'qos-pin':[
  'Which release lock file records the selected compiler revision?',
  'Explain the release-pinning operation for a clean compiler checkout.',
  'Does --pin record a compiler commit or an arbitrary Python package version?',
  'Describe what must be clean when pinning the language dependency for a release.',
  'A teammate thinks the pin operation configures physical GPIO pins. Explain its actual purpose.'],
 'qos-platform':[
  'Name a supported non-Linux host architecture/platform combination for the portable system.',
  'Can the portable host be built for FreeBSD on arm64?',
  'Explain the documented FreeBSD support without assuming a live deployment.',
  'Does supporting FreeBSD mean that Linux hosts are unsupported?',
  'Give a concise statement of portable host support that includes Linux and FreeBSD.'],
 'actor-static':[
  'Describe fixed actor mailbox rings and their capacity behavior.',
  'What growth rule applies when a dynamic sender-channel ring fills?',
  'Why is a static actor ring useful when a capacity bound matters?',
  'Do fixed mailbox rings silently allocate larger buffers when full?',
  'Explain where extra dynamic-ring capacity comes from.'],
 'actor-default':[
  'Name the fixed ring capacity chosen by the convenience actor-spawn operation.',
  'Does using plain spawn automatically request a dynamically growing mailbox?',
  'How many messages fit a default fixed sender-channel ring?',
  'State the capacity policy chosen by ordinary actor creation.',
  'A service needs growth during bursts. Explain why the default spawn policy does not promise that.'],
 'send-result':[
  'Does a successful actor send confirm queuing or completion of receiver work?',
  'List the result type and two refusal reasons for mailbox admission.',
  'Explain why Ok from send is not proof that the receiver has processed the request.',
  'How can a sender distinguish a dead target from successful queuing?',
  'A caller ignores send and assumes delivery always works. Explain the missing error handling.'],
 'send-retry':[
  'Should reliable actor admission retry a target that has already exited?',
  'Explain why the mailbox-full case yields and retries, while permanent refusal returns.',
  'State the retry boundary of the reliable mailbox admission helper.',
  'Which actor admission refusal is considered transient by sendSureResult?',
  'A retry loop treats every failure as mailbox pressure. Correct that policy.'],
 'call-wire':[
  'How is a reply associated with the particular actor request that caused it?',
  'Name the fields of a correlated request and the corresponding service reply.',
  'Why is a fresh request identifier included alongside the caller address?',
  'Can any queued message be consumed as the result of a correlated service call?',
  'Explain how matching the service and request identifier avoids accepting an unrelated reply.'],
 'call-death':[
  'Explain the error returned when a service exits while a correlated caller is waiting.',
  'Why does a death-aware actor request avoid leaving the caller blocked indefinitely?',
  'Which refusal string indicates the target service is no longer alive?',
  'Is service death treated as a successful empty response?',
  'A client waits forever after server termination. Describe the death-aware call contract it needs.'],
 'json-trailing':[
  'Is the JSON reader a prefix parser or a whole-input parser?',
  'Explain the error behavior for non-whitespace following a complete JSON document.',
  'What does the parser do after reading a JSON value but before returning it?',
  'Can a second arbitrary token follow a JSON value without rejection?',
  'A JSON reader silently drops an extra suffix. Describe the Sol library contract it violates.'],
 'json-duplicates':[
  'Explain lookup behavior when an object contains the same JSON key twice.',
  'Are repeated JSON object members retained in their original order?',
  'Does object lookup prefer the earliest or latest repeated key?',
  'What information survives parsing an object with duplicate member names?',
  'A teammate expects last-key-wins object access. Describe the library behavior.'],
 'checked-parse':[
  'What result shape should checked decimal conversion return for malformed text?',
  'Does checked string-to-integer conversion validate the complete input?',
  'How are integer-range failures reported by the checked string conversion?',
  'Contrast a checked decimal read with accepting only an initial numeric prefix.',
  'A decimal input ends in letters. Explain why the checked conversion refuses it.']}
  result=[]
  for row in old:
    key=row['id'].removeprefix('teach-')
    if row.get('program') or key not in variants:continue
    for i,q in enumerate(variants[key]):
      answer=row['answer']
      # Neutralize answers whose original introductory No depends on a question.
      if key in ('qos-platform','json-trailing','checked-parse') and answer.startswith('No. '):answer=answer[4:]
      if key=='json-trailing':answer='The Sol JSON parser validates the whole document and returns Err for trailing non-whitespace characters after a value.'
      if key=='sol-launch' and i in (0,3):
        filename='demo.sol' if i==0 else 'scripts/example.sol'
        answer=f'Run `./fpr sol {filename}`. The sol subcommand selects the hosted scripting profile.'
      result.append({**row,'id':f'expanded-fact-{key}-{i}','question':q,'answer':answer,
                     'category':'domain_contract','verification':'source-reviewed contract; no claim of executing actors or OS code'})
  return result


def source_for_builtin(snapshot, old, needle):
  path='fprisc/compiler/Sol/Preamble.hs'
  full=snapshot/path
  lines=full.read_text().splitlines()
  line=next((i+1,text) for i,text in enumerate(lines) if needle in text)
  original=next(s for r in old for s in r['sources'] if s['path']==path)
  return {**original,'sha256':file_hash(full),'evidence':line[1],'line_start':line[0],'line_end':line[0]}


def program_specs(seed):
  rng=random.Random(seed)
  specs=[]
  def add(family,expr,expected,goal,needle,explanation,wrong=None,wrong_expected=None):
    specs.append({'family':family,'expression':expr,'stdout':str(expected),'goal':goal,'needle':needle,
                  'explanation':explanation,'wrong':wrong,'wrong_stdout':None if wrong_expected is None else str(wrong_expected)})
  for j in range(8):
    n=11+3*j; lo=-17+j; hi=lo+9; x=hi+5 if j%2 else lo-6
    word=rng.choice(['lantern','sequoias','cobaltine','woodlands'])
    i=j%3; end=i+3
    xs=[n,n+2,n+5,n+9]; literal='['+', '.join(map(str,xs))+']'
    quoted=json.dumps(word)
    add('clamp',f'Numeric.clamp ({lo}) {hi} ({x})',min(hi,max(lo,x)),f'clamp {x} into the inclusive interval {lo} through {hi}',
        'clamp = fn lo hi a','Numeric.clamp takes the lower bound, upper bound, then the value.',
        f'Numeric.clamp {hi} ({lo}) ({x})',min(lo,max(hi,x)))
    add('at',f'Str.at {quoted} {i}',ord(word[i]),f'read the byte code at zero-based offset {i} of {quoted}',
        'at = fn s i','Str.at uses a zero-based byte offset; it returns the byte code, not a one-character string.',
        f'Str.at {quoted} {i+1}',ord(word[i+1]))
    add('slice',f'Str.slice {quoted} {i} {end}',word[i:end],f'extract bytes in the half-open interval [{i}, {end}) from {quoted}',
        'slice = fn s i j','The final slice argument is the exclusive end offset.',f'Str.slice {quoted} {i} {end+1}',word[i:end+1])
    add('sub',f'Str.sub {quoted} {i+1} 2',word[i+1:i+3],f'take two bytes starting at offset {i+1} in {quoted}',
        'sub = fn s i n','Str.sub takes a starting offset and a count.',f'Str.sub {quoted} {i+1} {i+3}',word[i+1:i+1+i+3])
    add('sum',f'List.sum {literal}',sum(xs),f'sum the integers {literal}','sum = fn xs','List.sum accumulates all list elements.')
    add('map',f'List.sum (List.map (fn v -> v * 3) {literal})',3*sum(xs),f'triple each integer in {literal} and sum the transformed list',
        'map = fn f xs','List.map applies its function to every element before summation.',f'List.sum (List.map (fn v -> v + 3) {literal})',sum(xs)+12)
    add('filter',f'List.sum (List.filter (fn v -> v >= {n+2}) {literal})',sum(xs[1:]),f'sum values at least {n+2} from {literal}',
        'filter = fn p xs','Use >= to retain values equal to the boundary as well as larger values.',f'List.sum (List.filter (fn v -> v > {n+2}) {literal})',sum(xs[2:]))
    add('fold',f'List.fold (fn acc v -> acc + v) 0 {literal}',sum(xs),f'compute the plain sum of {literal} using List.fold',
        'fold = fn f z xs','The initial accumulator for an ordinary sum is zero.',f'List.fold (fn acc v -> acc + v) 7 {literal}',sum(xs)+7)
    add('range',f'List.sum (List.range {n} {n+3})',sum(range(n,n+4)),f'sum all integers from {n} through {n+3}, including both endpoints',
        'range = fn lo hi','List.range includes its upper endpoint.',f'List.sum (List.range {n} {n+4})',sum(range(n,n+5)))
    add('take',f'List.sum (List.take 2 {literal})',sum(xs[:2]),f'sum only the first two elements of {literal}',
        'take = fn n xs','List.take retains the leading elements; it does not discard them.',f'List.sum (List.drop 2 {literal})',sum(xs[2:]))
    add('drop',f'List.sum (List.drop 1 {literal})',sum(xs[1:]),f'discard the first element of {literal} and sum what remains',
        'drop = fn n xs','List.drop removes the requested prefix.',f'List.sum (List.take 1 {literal})',xs[0])
    add('reverse',f'(List.rev {literal}) ! 0',xs[-1],f'reverse {literal} and select its first element',
        'rev = fn xs','Index zero selects the first element of the reversed list.',f'(List.rev {literal}) ! 1',xs[-2])
    add('abs',f'Numeric.abs ({-n})',n,f'compute the absolute value of {-n}','abs = fn a','A negative input becomes its positive magnitude.')
    add('min',f'Numeric.min {n+8} {n}',n,f'choose the smaller of {n+8} and {n}','min = fn a b','Numeric.min selects the smaller argument.',f'Numeric.max {n+8} {n}',n+8)
    add('max',f'Numeric.max {n} {n+8}',n+8,f'choose the larger of {n} and {n+8}','max = fn a b','Numeric.max selects the larger argument.')
    add('length',f'Str.len {quoted}',len(word),f'count bytes in the ASCII string {quoted}','len = fn s','All characters in this example are ASCII, so byte length equals character count.')
    add('list-length',f'List.len {literal}',len(xs),f'count elements in {literal}','len = fn xs','List.len counts elements rather than summing their values.')
    add('missing',f'Str.find 126 {quoted}',-1,f'find the byte code 126 in {quoted}','find = fn c s','A missing byte returns -1; zero is a valid matching offset.')
  return specs


def executable_candidates(snapshot, old, seed):
  rows=[]
  specs=program_specs(seed)
  for index,spec in enumerate(specs):
    source=source_for_builtin(snapshot,old,spec['needle'])
    expr=spec['expression']; value=spec['stdout']; goal=spec['goal']
    direct=f'> print (str ({expr})).\n'
    bound=f'result = {expr}.\n> print (str result).\n'
    program={'code':direct,'stdout':value+'\n','exit_code':0}
    base={'path':source['path'],'chunk':0,'style':'verified','round':0,'sources':[source],
          'expression':expr,'family':spec['family'],'verification':'authored specification; target execution required'}
    prediction_templates=[
      'Determine the exact line sent to stdout by this hosted script. Reply with the line only.',
      'Evaluate this script mentally and state its printed value without rewriting the program.',
      'Trace the operations and report the final text written by print.',
      'Give just the resulting stdout line for this small program.']
    rows.append({**base,'id':f'expanded-output-{index}','category':'output_prediction',
                 'question':prediction_templates[index%4]+'\n```sol\n'+direct+'```',
                 'answer':value,'program':program})
    code_templates=[
      f'Provide a runnable Sol file to {goal}. Print the result and include no unrelated operations.',
      f'Use Sol to {goal}; show a complete program with explicit printing.',
      f'Implement this task as a short hosted script: {goal}.',
      f'I need Sol code that will {goal}. Bind the value to result and print it.']
    rows.append({**base,'id':f'expanded-code-{index}','category':'code_generation',
                 'question':code_templates[index%4], 'answer':'```sol\n'+bound+'```',
                 'program':{'code':bound,'stdout':value+'\n','exit_code':0}})
    rows.append({**base,'id':f'expanded-explain-{index}','category':'code_explanation',
                 'question':f'Explain how to {goal} using the expression `{expr}`.',
                 'answer':spec['explanation']+' The result is '+value+'.', 'program':program})
    if spec['wrong']:
      broken=f'> print (str ({spec["wrong"]})).\n'
      rows.append({**base,'id':f'expanded-debug-{index}','category':'debugging',
                   'question':f'I want to {goal}. This script instead prints {spec["wrong_stdout"]}. Correct the mistake and explain it.\n```sol\n'+broken+'```',
                   'answer':spec['explanation']+'\n```sol\n'+direct+'```\nThe corrected stdout is '+value+'.',
                   'program':program,'debug_program':{'code':broken,'stdout':spec['wrong_stdout']+'\n','exit_code':0}})
  for index,spec in enumerate(specs[:16]):
    source=source_for_builtin(snapshot,old,spec['needle'])
    expr=spec['expression'];code=f'result = {expr}.\n> print (str result).\n'
    rows.append({'id':f'expanded-unbound-{index}','path':source['path'],'chunk':0,'style':'verified','round':0,
       'category':'debugging','family':'unbound-result','sources':[source],'expression':expr,
       'question':f'This Sol file refers to result before defining it. Add a definition that will {spec["goal"]}.\n```sol\n> print (str result).\n```',
       'answer':'Bind result before printing it.\n```sol\n'+code+'```',
       'program':{'code':code,'stdout':spec['stdout']+'\n','exit_code':0},
       'debug_program':{'code':'> print (str result).\n','stdout':'','exit_code':1},
       'verification':'authored repair; compiler failure and corrected behavior must both be checked'})
  return rows


def build(args):
  if args.output.exists():raise ValueError('Use a new curriculum directory')
  old=read_rows(args.original/'verified_train.jsonl')
  benchmark=read_rows(args.original/'benchmark.jsonl')
  base=json.loads(args.base_splits.read_text())
  qa=read_rows(args.qa)
  candidates=fact_candidates(old)+executable_candidates(args.snapshot,old,args.seed)
  kept,rejected,manifest=screen_extension(base,old,candidates,qa,benchmark)
  args.output.mkdir(parents=True)
  write_rows(args.output/'verified_train.jsonl',kept)
  write_rows(args.output/'rejected.jsonl',rejected)
  # Benchmark bytes and original reference evidence remain immutable.
  (args.output/'benchmark.jsonl').write_bytes((args.original/'benchmark.jsonl').read_bytes())
  (args.output/'splits.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  summary={**manifest['extension_counts'],'benchmark_sha256':file_hash(args.output/'benchmark.jsonl'),
           'base_split_hash':base['hash'],'expanded_split_hash':manifest['hash'],
           'distinct_reference_programs':len({r['program']['code'] for r in kept if r.get('program')}),
           'generator_sha256':file_hash(Path(__file__)),'seed':args.seed}
  (args.output/'build.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


def verify(args):
  rows=read_rows(args.data/'verified_train.jsonl')
  benchmark=read_rows(args.data/'benchmark.jsonl')
  roots={'fprisc':str(args.snapshot/'fprisc'),'qos':str(args.snapshot/'qos')}
  for row in rows+benchmark:source_check(row,roots)
  results=[]; cache={}
  for row in rows:
    for field in ('program','debug_program'):
      if field not in row:continue
      p=row[field];key=digest(p)
      if key not in cache:
        with tempfile.TemporaryDirectory(prefix='qos-ai-expanded-check-') as directory:
          script=Path(directory)/'example.sol';script.write_text(p['code'])
          env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=roots['fprisc'])
          run=subprocess.run([str(args.fpr.resolve()),'sol',str(script)],cwd=directory,env=env,capture_output=True,text=True,timeout=30)
          passed=run.returncode==p['exit_code'] and run.stdout.strip()==p['stdout'].strip()
          cache[key]={'passed':passed,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
      result={'id':row['id'] if field=='program' else row['id']+':broken','fixture':field,**cache[key]}
      results.append(result)
      if not result['passed']:print(json.dumps(result),flush=True)
  proof={'verified_data_hash':rows_fingerprint(rows),'benchmark_data_hash':rows_fingerprint(benchmark),
         'host':platform.node(),'machine':platform.machine(),
         'fpr_sha256':file_hash(args.fpr),'passed':sum(r['passed'] for r in results),'total':len(results),
         'distinct_executions':len(cache),'results':results}
  (args.data/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  if not results or proof['passed'] != proof['total']:raise SystemExit(1)
  manifest=json.loads((args.data/'splits.json').read_text())
  validate_manifest(manifest,verified=rows,benchmark=benchmark)
  if any(row_split(r,manifest)!='train' for r in rows):raise ValueError('Non-training row survived preparation')
  write_rows(args.data/'verified_train_ready.jsonl',rows)
  print(json.dumps({k:v for k,v in proof.items() if k!='results'},indent=2))


def main():
  p=argparse.ArgumentParser(description=__doc__)
  sub=p.add_subparsers(dest='command',required=True)
  b=sub.add_parser('build');b.set_defaults(fn=build)
  b.add_argument('--original',type=Path,default=Path('qos_ai_benchmark'))
  b.add_argument('--base-splits',required=True,type=Path)
  b.add_argument('--qa',required=True,type=Path)
  b.add_argument('--snapshot',required=True,type=Path)
  b.add_argument('--output',required=True,type=Path)
  b.add_argument('--seed',type=int,default=29)
  v=sub.add_parser('verify');v.set_defaults(fn=verify)
  v.add_argument('--data',required=True,type=Path)
  v.add_argument('--snapshot',required=True,type=Path)
  v.add_argument('--fpr',required=True,type=Path)
  args=p.parse_args();args.fn(args)


if __name__=='__main__':main()
