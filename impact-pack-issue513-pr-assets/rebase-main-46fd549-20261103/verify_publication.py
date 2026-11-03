from pathlib import Path
from urllib.request import urlopen,Request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from datetime import datetime,timezone
import json,subprocess,re,hashlib,time,os
P=Path(__file__).resolve().parent;ROOT=P.parents[1];R=ROOT/'impact-pack-pr-issue513-fixed-seed-controls';B=ROOT/'impact-pack-issue513-evidence-publish';REPO='ltdrdata/ComfyUI-Impact-Pack'
meta=json.loads((P/'publication.json').read_text());body=(P/'pr-body.md').read_text();env={k:v for k,v in os.environ.items() if k not in ('GITHUB_TOKEN','GH_TOKEN')}
def gh(*args):return subprocess.check_output(['gh',*args],env=env,text=True)
pr=json.loads(gh('api','repos/'+REPO+'/pulls/1271'));assert pr['head']['sha']==meta['code_head'] and pr['base']['sha']==meta['base'] and pr['commits']==1 and pr['state']=='open' and not pr['draft'];assert pr['body']==body
files=gh('api','repos/'+REPO+'/pulls/1271/files?per_page=100','--jq','.[].filename').splitlines();expected=subprocess.check_output(['git','diff','--name-only','origin/Main...HEAD'],cwd=R,text=True).splitlines();assert sorted(files)==sorted(expected) and len(files)==9
query='query { repository(owner:"ltdrdata",name:"ComfyUI-Impact-Pack") { pullRequest(number:1271) { closingIssuesReferences(first:100) { nodes { number } } } issue(number:513) { state } } }'
refs=json.loads(gh('api','graphql','-f','query='+query))['data']['repository'];assert not refs['pullRequest']['closingIssuesReferences']['nodes'] and refs['issue']['state']=='OPEN'
request=P/'markdown-request.json';request.write_text(json.dumps({'text':body,'mode':'gfm','context':REPO}));render=gh('api','markdown','--input',str(request));(P/'rendered-body.html').write_text(render)
class Tags(HTMLParser):
 def __init__(self):super().__init__();self.details=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  if tag=='details':self.details.append(dict(attrs))
  if tag=='img':self.images.append(dict(attrs))
tags=Tags();tags.feed(render);assert len(tags.details)==3 and all('open' not in a for a in tags.details) and len(tags.images)==18
remote=json.loads(gh('api',f'repos/{REPO}/git/trees/{meta["evidence_commit"]}?recursive=1'));assert not remote.get('truncated');remote_blobs={x['path']:x['sha'] for x in remote['tree'] if x['type']=='blob'}
local=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=B,text=True).splitlines();assert remote_blobs=={line.split('\t',1)[1]:line.split()[2] for line in local}
urls=re.findall(r'!\[[^\]]*\]\((https://raw\.githubusercontent\.com/[^)]+)\)',body);assert len(urls)==len(set(urls))==18
def verify(url):
 rel=url.split('/'+meta['evidence_commit']+'/',1)[1];expected=hashlib.sha256((B/rel).read_bytes()).hexdigest()
 for attempt in range(4):
  try:
   with urlopen(Request(url,headers={'User-Agent':'PR-evidence-verification'}),timeout=30) as response:data=response.read()
   assert data.startswith(b'\x89PNG\r\n\x1a\n') and hashlib.sha256(data).hexdigest()==expected
   return {'path':rel,'sha256':expected,'bytes':len(data)}
  except OSError:
   if attempt==3:raise
   time.sleep(1)
with ThreadPoolExecutor(max_workers=6) as pool:images=list(pool.map(verify,urls))
receipt={'verified_utc':datetime.now(timezone.utc).isoformat(),'pr':pr['html_url'],'head':pr['head']['sha'],'base':pr['base']['sha'],'commits':1,'files':files,'body_matches':True,'collapsed_details':3,'inline_images':18,'public_png_hashes':images,'evidence_commit':meta['evidence_commit'],'evidence_tree_exact':True,'issue_513_state':refs['issue']['state'],'closing_issues':[],'mergeable':pr['mergeable'],'mergeable_state':pr['mergeable_state']}
(P/'remote-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');(ROOT/'impact-pack-issue513-pr-published.md').write_text(body)
print('PR head/base/body, one commit, nine files, three collapsed blocks, 18 public PNG hashes, exact evidence tree and open issue #513 verified.')
