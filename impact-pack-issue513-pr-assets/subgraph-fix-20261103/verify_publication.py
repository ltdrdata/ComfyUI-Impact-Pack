from pathlib import Path
from urllib.request import urlopen,Request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json,subprocess,re,hashlib,time,os
P=Path(__file__).resolve().parent;ROOT=P.parents[1];R=ROOT/'impact-pack-pr-issue513-fixed-seed-controls';B=ROOT/'impact-pack-issue513-evidence-publish'
REMOTE='ltdrdata/ComfyUI-Impact-Pack';meta=json.loads((P/'publication.json').read_text());body=(P/'pr-body.md').read_text()
pr=json.loads(subprocess.check_output(['gh','pr','view','1271','--repo',REMOTE,'--json','body,headRefOid,baseRefName,commits,files,state,isDraft'],text=True))
assert pr['body']==body and pr['headRefOid']==meta['code_head'] and pr['baseRefName']=='Main' and pr['state']=='OPEN' and not pr['isDraft'] and len(pr['commits'])==1
assert sorted(f['path'] for f in pr['files'])==sorted(subprocess.check_output(['git','diff','--name-only','origin/Main...HEAD'],cwd=R,text=True).splitlines())
query='query { repository(owner:"ltdrdata",name:"ComfyUI-Impact-Pack") { pullRequest(number:1271) { closingIssuesReferences(first:100) { nodes { number } } } issue(number:513) { state } } }'
refs=json.loads(subprocess.check_output(['gh','api','graphql','-f','query='+query],text=True))['data']['repository'];assert not refs['pullRequest']['closingIssuesReferences']['nodes'] and refs['issue']['state']=='OPEN'
request=P/'markdown-request.json';request.write_text(json.dumps({'text':body,'mode':'gfm','context':REMOTE}))
env={k:v for k,v in os.environ.items() if k not in ('GITHUB_TOKEN','GH_TOKEN')}
render=subprocess.check_output(['gh','api','markdown','--input',str(request)],env=env,text=True);(P/'rendered-body.html').write_text(render)
class Tags(HTMLParser):
 def __init__(self):super().__init__();self.details=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  if tag=='details':self.details.append(dict(attrs))
  if tag=='img':self.images.append(dict(attrs))
tags=Tags();tags.feed(render);assert len(tags.details)==3 and all('open' not in d for d in tags.details) and len(tags.images)==18
urls=re.findall(r'!\[[^\]]*\]\((https://raw\.githubusercontent\.com/[^)]+)\)',body);assert len(urls)==18 and len(set(urls))==18
# Git tree objects must exactly match the local evidence bundle.
remote_tree=json.loads(subprocess.check_output(['gh','api',f'repos/{REMOTE}/git/trees/{meta["evidence_commit"]}?recursive=1'],text=True));assert not remote_tree.get('truncated')
remote_blobs={x['path']:x['sha'] for x in remote_tree['tree'] if x['type']=='blob'}
local=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=B,text=True).splitlines();assert remote_blobs=={line.split('\t',1)[1]:line.split()[2] for line in local}
def verify(url):
 relative=url.split('/'+meta['evidence_commit']+'/',1)[1];expected=hashlib.sha256((B/relative).read_bytes()).hexdigest()
 for attempt in range(4):
  try:
   with urlopen(Request(url,headers={'User-Agent':'PR-evidence-verification'}),timeout=30) as response:data=response.read()
   assert data.startswith(b'\x89PNG\r\n\x1a\n') and hashlib.sha256(data).hexdigest()==expected
   return {'path':relative,'sha256':expected,'bytes':len(data)}
  except OSError:
   if attempt==3:raise
   time.sleep(1)
with ThreadPoolExecutor(max_workers=6) as pool:images=list(pool.map(verify,urls))
receipt={'head':pr['headRefOid'],'base':meta['base'],'commits':len(pr['commits']),'files':[f['path'] for f in pr['files']],'body_matches':True,'details_closed_by_default':3,'rendered_inline_images':18,'public_png_sha256_verified':images,'evidence_tree_exact':True,'closing_issues':refs['pullRequest']['closingIssuesReferences']['nodes'],'issue_513_state':refs['issue']['state']}
(P/'remote-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Remote PR head/body, one commit, 3 collapsed blocks, 18 image hashes, evidence tree and open issue #513 verified.')
