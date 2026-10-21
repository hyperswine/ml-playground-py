"""Conservative pure Sol subset for coding diagnostics, with structural contracts.

This is a restricted evaluator, not a general Sol validator or OS sandbox. Only
reviewed numeric/list/string primitives and one print effect are admitted.
"""
import json
import re

FUNCTIONS={'str','List.sum','List.map','List.filter','List.range','List.take','List.drop','List.rev','List.len',
           'Str.len','Str.at','Str.slice','Str.sub','Str.find','Numeric.clamp','Numeric.min','Numeric.max','Numeric.abs'}
LIST_BUILDERS={'List.map','List.filter','List.range','List.take','List.drop','List.rev'}
PRECEDENCE={'==':10,'>=':10,'<=':10,'>':10,'<':10,'+':20,'-':20,'*':30,'/':30,'!':40}


class Unsupported(ValueError):pass


def lex(code):
  if len(code)>6000:raise Unsupported('program exceeds diagnostic size limit')
  result=[];i=0
  while i<len(code):
    if code[i].isspace():i+=1;continue
    if code[i]=='#':
      end=code.find('\n',i);i=len(code) if end<0 else end+1;continue
    if code[i]=='"':
      try:value,end=json.JSONDecoder().raw_decode(code[i:])
      except ValueError as e:raise Unsupported('invalid string literal') from e
      if not isinstance(value,str) or len(value)>128 or any(c in value for c in '{}') or not value.isascii():
        raise Unsupported('only short ASCII strings without interpolation are admitted')
      result.append(('string',value));i+=end;continue
    m=re.match(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|->|>=|<=|==|[()\[\].,=+*/!<>-]',code[i:])
    if not m:raise Unsupported('unsupported token')
    token=m[0];i+=len(token)
    if token.isdigit():
      if int(token)>1000:raise Unsupported('integer outside diagnostic bound')
      result.append(('int',int(token)))
    else:result.append(token)
  if len(result)>1200:raise Unsupported('too many tokens')
  return result


class Parser:
  def __init__(self,code):self.tokens=lex(code);self.i=0;self.env={};self.locals=set();self.depth=0
  def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else None
  def pop(self):
    t=self.peek()
    if t is None:raise Unsupported('unexpected end')
    self.i+=1;return t
  def expect(self,t):
    if self.pop()!=t:raise Unsupported('unexpected syntax')
  def starts_atom(self,t):return isinstance(t,tuple) or t in ('(','[','fn') or isinstance(t,str) and bool(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',t))
  def atom(self):
    t=self.pop()
    if isinstance(t,tuple):return t
    if t=='-':return ('neg',self.atom())
    if t=='(':
      value=self.expr();self.expect(')');return value
    if t=='[':
      values=[]
      if self.peek()!=']':
        while True:
          values.append(self.expr())
          if len(values)>64:raise Unsupported('list literal too large')
          if self.peek()!=',':break
          self.pop()
      self.expect(']');return ('list',tuple(values))
    if t=='fn':
      names=[]
      while self.peek()!='->':
        n=self.pop()
        if not isinstance(n,str) or not re.fullmatch(r'[a-z][A-Za-z_0-9]*',n):raise Unsupported('invalid lambda parameter')
        names.append(n)
        if len(names)>2:raise Unsupported('lambda arity outside subset')
      if not names or len(set(names))!=len(names):raise Unsupported('invalid lambda parameters')
      self.pop();previous=self.locals.copy();self.locals.update(names);body=self.expr();self.locals=previous
      return ('lambda',tuple(names),body)
    if isinstance(t,str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',t):
      if t in ('List','Str','Numeric') and self.peek()=='.':
        self.pop();t+='.'+self.pop()
      if t in self.locals:return ('var',t)
      if t in self.env:return self.env[t]
      if t in FUNCTIONS:return ('name',t)
      raise Unsupported('unknown/free function or variable: '+t)
    raise Unsupported('unsupported expression')
  def expr(self,minimum=0):
    self.depth+=1
    if self.depth>24:raise Unsupported('expression nesting too deep')
    left=self.atom()
    while True:
      token=self.peek()
      if self.starts_atom(token) and 50>=minimum:
        left=('app',left,self.expr(51));continue
      if isinstance(token,str) and token in PRECEDENCE and PRECEDENCE[token]>=minimum:
        op=self.pop();left=('bin',op,left,self.expr(PRECEDENCE[op]+1));continue
      break
    self.depth-=1;return left
  def expression(self):
    node=self.expr()
    if self.peek()=='.':self.pop()
    if self.peek() is not None:raise Unsupported('trailing expression text')
    validate_tree(node);return normalize(node)
  def program(self):
    prints=[]
    while self.peek() is not None:
      if self.peek()=='>':
        self.pop();self.expect('print');value=self.expr();self.expect('.');prints.append(value)
        if len(prints)>1:raise Unsupported('only one print is admitted')
      else:
        name=self.pop()
        if not isinstance(name,str) or not re.fullmatch(r'[a-z][A-Za-z_0-9]*',name) or name in FUNCTIONS:
          raise Unsupported('unsupported top-level declaration')
        if name in self.env:raise Unsupported('redefinition outside subset')
        self.expect('=');value=self.expr();self.expect('.');validate_tree(value);self.env[name]=value
        if len(self.env)>16:raise Unsupported('too many bindings')
    if len(prints)!=1:raise Unsupported('program must explicitly print once')
    value=prints[0]
    if value[:1]==('app',) and value[1]==('name','str'):value=value[2]
    validate_tree(value);return normalize(value)


def spine(node):
  args=[]
  while node[0]=='app':args.append(node[2]);node=node[1]
  return node,list(reversed(args))


def children(node):
  if node[0]=='app':return [node[1],node[2]]
  if node[0]=='bin':return [node[2],node[3]]
  if node[0]=='neg':return [node[1]]
  if node[0]=='list':return list(node[1])
  if node[0]=='lambda':return [node[2]]
  return []


def number(node):
  if node[0]=='int':return node[1]
  if node[0]=='neg':return -number(node[1])
  raise Unsupported('range bounds must be constant integers')


def validate_tree(node,in_lambda=False):
  kind=node[0]
  if in_lambda and (kind=='list' or kind=='name' and node[1] in LIST_BUILDERS):
    raise Unsupported('list construction inside lambdas is outside the bounded subset')
  if kind=='app':
    function,args=spine(node)
    if in_lambda and function[0]=='name' and function[1] in LIST_BUILDERS:
      raise Unsupported('list construction inside lambdas is outside the bounded subset')
    if function==('name','List.range'):
      if len(args)!=2 or abs(number(args[1])-number(args[0]))>64:raise Unsupported('range outside diagnostic bound')
    else:validate_tree(function,in_lambda)
    for arg in args:validate_tree(arg,in_lambda)
  elif kind=='name' and node[1]=='List.range':raise Unsupported('partial range outside bounded subset')
  else:
    for child in children(node):validate_tree(child,in_lambda or kind=='lambda')


def normalize(node,names=None):
  names=names or {}
  if node[0]=='var':return ('var',names.get(node[1],node[1]))
  if node[0]=='lambda':
    local=dict(names);start=len(local)
    for i,name in enumerate(node[1]):local[name]='arg'+str(start+i)
    return ('lambda',tuple(local[n] for n in node[1]),normalize(node[2],local))
  if node[0]=='neg' and node[1][0]=='int':return ('int',-node[1][1])
  if node[0]=='app':return ('app',normalize(node[1],names),normalize(node[2],names))
  if node[0]=='bin':
    left,right=normalize(node[2],names),normalize(node[3],names)
    if node[1] in ('+','*'):left,right=sorted((left,right),key=repr)
    return ('bin',node[1],left,right)
  if node[0]=='list':return ('list',tuple(normalize(v,names) for v in node[1]))
  return node


def extract(answer):
  blocks=re.findall(r'```(?:sol|fprisc)?\s*\n(.*?)```',answer,re.S)
  if len(blocks)>1:raise Unsupported('ambiguous multiple code blocks')
  if blocks:
    text=blocks[0].strip()
    try:return text,Parser(text).program(),'program'
    except Unsupported:
      ast=Parser(text).expression();return f'> print (str ({text})).\n',ast,'expression wrapped only for printing'
  try:return answer.strip(),Parser(answer).program(),'program'
  except Unsupported:pass
  snippets=re.findall(r'`([^`\n]+)`',answer)
  parsed=[]
  for text in snippets:
    try:parsed.append((text,Parser(text).expression()))
    except Unsupported:pass
  if len(parsed)!=1:raise Unsupported('no unambiguous supported program/expression')
  text,ast=parsed[0];return f'> print (str ({text})).\n',ast,'expression wrapped only for printing'
