"""Real renders of a promoted fixed seed and an independent increment sampler."""
from pathlib import Path
from datetime import datetime,timezone
from argparse import Namespace
import hashlib,json,os,subprocess,sys,time
from urllib.request import urlopen
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sampling';OUT.mkdir(exist_ok=True)
REPO=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls');COMFY=Path('/mnt/teratera/git/ComfyUI')
sys.path.insert(0,str(REPO/'tests'))
from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow,full_zoom,show_node,rendered
from test_e2e_wildcard_cache import convert_to_subgraph,enter_subgraphs,interior_mode_point,control_timing,cached
PY='/home/rho/venv/bin/python';PORT=18414;URL=f'http://127.0.0.1:{PORT}'
parent_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
base=subprocess.check_output(['git','rev-parse','b1ee82d'],cwd=REPO,text=True).strip()
candidate=(REPO/'js/impact-pack.js').read_bytes();baseline=subprocess.check_output(['git','show',base+':js/impact-pack.js'],cwd=REPO)
source=ROOT.parent/'fixed-mode/seed-refresh/independent-sampler/two-branch-workflow.json'
workflow=OUT/'promoted-two-branch-workflow.json';reports={};host_id=None

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fit(ui):
 ui.page.mouse.click(110,170)
 ui.page.get_by_role('button',name='Fit View (.)',exact=True).click();rendered(ui)

def state(ui,host):
 return ui.app.evaluate('''(app,id)=>{
  const snapshot=n=>({id:String(n.id),type:n.type,widgets:(n.widgets??[]).filter(w=>!w.name.startsWith('$$')).map(w=>({name:w.name,value:w.value??null}))});
  const host=app.graph.getNodeById(id);
  return {root:Object.fromEntries(app.graph._nodes.map(n=>[String(n.id),snapshot(n)])),interior:snapshot(host.subgraph._nodes.find(n=>n.type==='ImpactWildcardEncode'))};
 }''',host)

def vals(node):return {w['name']:w['value'] for w in node['widgets']}
def ancestry(prompt,target):
 seen=set()
 def visit(n):
  for v in prompt[n]['inputs'].values():
   if isinstance(v,list) and len(v)==2 and str(v[0]) in prompt and str(v[0]) not in seen:
    seen.add(str(v[0]));visit(str(v[0]))
 visit(target);return seen

for phase,data in [('candidate',candidate),('baseline',baseline)]:
 profile='sampling-'+phase;folder=OUT/phase;folder.mkdir(exist_ok=True)
 with (OUT/('server-'+phase+'.log')).open('w') as log:
  server=subprocess.Popen([PY,ROOT/'serve.py','--profile',profile,'--gpu','--port',str(PORT)],stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(120):
    assert server.poll() is None
    try:
     with urlopen(URL+'/object_info',timeout=1) as r:
      if 'ImpactWildcardEncode' in json.load(r):break
    except OSError:time.sleep(.5)
   else:raise RuntimeError('Server startup')
   pack=ROOT/('runtime-'+profile)/'custom_nodes/comfyui-impact-pack';(pack/'js/impact-pack.js').write_bytes(data)
   args=Namespace(base_url=URL,browser_channel='chrome',headed=False,timeout=900,artifacts=folder)
   with browser_session(args) as ui:
    ui.page.set_viewport_size({'width':2400,'height':1800});ui.page.keyboard.press('Escape');ui.new_workflow()
    if not workflow.exists():
     original=json.loads(source.read_text());ui.load_workflow(source,len(original['nodes']));full_zoom(ui);show_node(ui,'10');ui.choose('ImpactWildcardEncode','mode','populate')
     host_id=convert_to_subgraph(ui,'10');show_node(ui,host_id);enter_subgraphs(ui,1)
     ui.page.mouse.click(*interior_mode_point(ui,'ImpactWildcardEncode'),button='right')
     ui.page.get_by_role('menuitem',name='Promote Widget: mode',exact=True).click()
     ui.page.get_by_test_id('subgraph-breadcrumb-back').click()
     ui.page.wait_for_function('app=>app.canvas.graph===app.rootGraph',arg=ui.app)
     show_node(ui,host_id)
     right=ui.app.evaluate('''(app,id)=>{const n=app.graph.getNodeById(id),m=n.widgets.find(w=>w.name==='mode');return app.canvasPosToClientPos([n.pos[0]+n.size[0]-25,n.pos[1]+(m.last_y??m.y)+10])}''',host_id)
     ui.page.keyboard.press('Escape');ui.page.mouse.click(*right,delay=100)
     ui.page.wait_for_function('''({app,id})=>app.graph.getNodeById(id).widgets.find(w=>w.name==='mode').value==='fixed' ''',arg={'app':ui.app,'id':host_id})
     export_workflow(ui,workflow);write(OUT/'host.json',{'host_id':host_id})
    else:host_id=json.loads((OUT/'host.json').read_text())['host_id']
    native=json.loads(workflow.read_text());ui.new_workflow();ui.load_workflow(workflow,len(native['nodes']));full_zoom(ui);control_timing(ui,'after')
    initial=state(ui,host_id);assert vals(initial['root'][host_id])=={'seed':513,'mode':'fixed'}
    assert vals(initial['interior'])['control_after_generate']=='increment'
    assert vals(initial['interior'])['mode']=='populate'
    rows=[]
    for q in range(1,4):
     if q==3:
      before=state(ui,host_id);export_workflow(ui,folder/'before-third-queue.json');ui.new_workflow();ui.load_workflow(folder/'before-third-queue.json',len(native['nodes']));full_zoom(ui)
      assert state(ui,host_id)==before
     fit(ui);entry=ui.run();write(folder/f'queue-{q}-history.json',entry)
     prompt=entry['prompt'][2];target=next(k for k,v in prompt.items() if v['class_type']=='ImpactWildcardEncode')
     seed=513 if phase=='candidate' else 512+q
     assert prompt[target]['inputs']['seed']==seed and prompt[target]['inputs']['mode']=='fixed'
     assert prompt['3']['inputs']['seed']==20261103 and prompt['17']['inputs']['seed']==9000+q
     dependencies=ancestry(prompt,'17');assert dependencies=={'4','5','7','16'}
     fixed_targets={target,'3','8','9'};cache=cached(entry)
     assert fixed_targets<=cache if phase=='candidate' and q>1 else fixed_targets.isdisjoint(cache)
     assert {'17','18','19'}.isdisjoint(cache)
     images=ui.images(entry);hashes={}
     for branch,node_id in [('fixed','9'),('independent','19')]:
      image=images[node_id][0];assert image.size==(512,512);image.save(folder/f'queue-{q}-{branch}-output.png');hashes[branch]=hashlib.sha256(image.tobytes()).hexdigest()
      filename=entry['outputs'][node_id]['images'][0]['filename']
      ui.page.wait_for_function('''({app,id,filename})=>app.graph.getNodeById(id).imgs?.some(img=>img.complete&&img.naturalWidth>0&&new URL(img.src).searchParams.get('filename')===filename)''',arg={'app':ui.app,'id':node_id,'filename':filename})
     current=state(ui,host_id);write(folder/f'queue-{q}-widget-state.json',current)
     assert vals(current['root'][host_id])=={'seed':513 if phase=='candidate' else seed+1,'mode':'fixed'}
     assert vals(current['root']['17'])['seed']==9001+q
     assert vals(current['interior'])['control_after_generate']=='increment'
     assert vals(current['interior'])['mode']=='populate'
     fit(ui);ui.page.screenshot(path=str(folder/f'queue-{q}-workflow.png'))
     regions=ui.app.evaluate('''(app,host)=>{
      const mark=(id,widget)=>{const n=app.graph.getNodeById(id),w=widget?n.widgets.find(w=>w.name===widget):null,y=w?(w.last_y??w.y):null;const a=app.canvasPosToClientPos([n.pos[0],n.pos[1]+(w?y-2:-30)]),b=app.canvasPosToClientPos([n.pos[0]+n.size[0],n.pos[1]+(w?y+22:n.size[1])]);return{node_id:id,widget:widget??null,x:a[0],y:a[1],width:b[0]-a[0],height:b[1]-a[1]}};
      return [mark(host,null),mark('3','seed'),mark('9',null),mark('17','seed'),mark('19',null)];
     }''',host_id);write(folder/f'queue-{q}-highlight-regions.json',regions)
     start=next(d['timestamp'] for e,d in entry['status']['messages'] if e=='execution_start')
     row={'queue':q,'prompt_id':entry['prompt'][1],'execution_start_ms':start,'fixed_execution_seed':seed,'fixed_ui_seed_after_queue':vals(current['root'][host_id])['seed'],'independent_execution_seed':9000+q,'independent_ui_seed_after_queue':9001+q,'fixed_branch_cached':fixed_targets<=cache,'independent_branch_cached':False,'independent_sampler_ancestors':sorted(dependencies),'cached_nodes':sorted(cache),'fixed_rgb_sha256':hashes['fixed'],'independent_rgb_sha256':hashes['independent'],'screenshot_sha256':sha(folder/f'queue-{q}-workflow.png'),'history_sha256':sha(folder/f'queue-{q}-history.json'),'native_reload_before_queue':q==3}
     rows.append(row);write(folder/'queues.json',rows);print(phase,q,row['fixed_execution_seed'],row['fixed_branch_cached'],flush=True)
    assert len({r['prompt_id'] for r in rows})==len({r['screenshot_sha256'] for r in rows})==3
    assert len({r['independent_rgb_sha256'] for r in rows})==3
    if phase=='candidate':assert len({r['fixed_rgb_sha256'] for r in rows})==1
    reports[phase]={'capture_parent_head':parent_head,'commit':base if phase=='baseline' else 'working-tree','source_js_sha256':hashlib.sha256(data).hexdigest(),'queues':rows,'native_widget_compatibility':True}
  finally:server.terminate();server.wait(timeout=20)
 reports[phase]['server_stopped']=True;write(folder/'result.json',reports[phase])
parity=[]
for q in range(1,4):
 def rgb(phase,branch):return np.asarray(Image.open(OUT/phase/f'queue-{q}-{branch}-output.png').convert('RGB'))
 diff=int(np.any(rgb('baseline','independent')!=rgb('candidate','independent'),axis=2).sum());assert diff==0
 if q==1:assert np.array_equal(rgb('baseline','fixed'),rgb('candidate','fixed'))
 parity.append({'queue':q,'independent_baseline_candidate_different_rgb_pixels':diff})
write(OUT/'comparison.json',{'successful':True,'capture_parent_head':parent_head,'candidate_source_js_sha256':hashlib.sha256(candidate).hexdigest(),'base':base,'host_id':host_id,'resolution':[512,512],'workflow_sha256':sha(workflow),'reports':reports,'independent_branch_pixel_parity':parity,'first_fixed_baseline_candidate_different_rgb_pixels':0,'completed_utc':datetime.now(timezone.utc).isoformat()})
print('Six actual promoted-seed captures and twelve output PNGs verified.',flush=True)
