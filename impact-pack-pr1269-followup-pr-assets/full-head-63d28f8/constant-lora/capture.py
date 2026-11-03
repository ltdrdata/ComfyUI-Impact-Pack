from argparse import Namespace
from pathlib import Path
from urllib.parse import urlsplit
import hashlib,json,os,sys
sys.path.insert(0,'/mnt/teratera/git/impact-pack-pr1269-followup-pr-assets/full-head-63d28f8/source/tests')
from e2e_ui import browser_session
from test_e2e_widget_catalog import full_zoom,pan,rendered,setting,show_node,export_workflow
from test_e2e_wildcard_cache import cached,input_slot,native_roundtrip
from wildcard_tutorial_cases import tutorial

out=Path(__file__).resolve().parent
args=Namespace(base_url=os.environ['IMPACT_E2E_BASE_URL'],browser_channel='chrome',headed=False,timeout=900,artifacts=out)
path,workflow,target=tutorial('/mnt/teratera/git/ComfyUI-extension-tutorials','lbw')
encoder=str(target['id']);sampler='3';output='9'
text='<lora:SD1.5/etc/add_detail:0.7:0.7:LBW=0,0,0,0,0,0,0,A,A,B,B,0,0,0,0,0,0;A=2.0;B=1.5>, a detailed photograph of a stone cottage beside a lake, pine trees, soft morning light'
edit=', red flowers beside the door'
events=[];requests=[];results=[]

def frame(payload):
 if isinstance(payload,str):
  message=json.loads(payload)
  if message['type'] in ['execution_start','execution_cached','executing','executed','execution_success']:events.append(message)

def bounds(ui):
 return ui.app.evaluate('''app=>app.graph._nodes.map(n=>({id:String(n.id),type:n.type,pos:app.canvasPosToClientPos(n.pos),size:Array.from(n.size),scale:app.canvas.ds.scale,widgets:(n.widgets??[]).map(w=>({name:w.name,value:w.value??null}))}))''')

def arrange(ui):
 ui.page.set_viewport_size({'width':3800,'height':2000});full_zoom(ui)
 b=bounds(ui);pan(ui,100-min(n['pos'][0] for n in b),140-min(n['pos'][1] for n in b))
 positions={'CheckpointLoaderSimple':(100,170),'PrimitiveStringMultiline':(100,440),'ImpactWildcardEncode':(580,170),'CLIPTextEncode':(580,730),'EmptyLatentImage':(1160,730),'KSampler':(1160,170),'VAEDecode':(1520,170),'SaveImage':(1770,170)}
 for n in bounds(ui):
  x,y=n['pos'];tx,ty=positions[n['type']]
  ui.page.mouse.move(x+70,y-15);ui.page.mouse.down();ui.page.mouse.move(tx+70,ty-15,steps=12);ui.page.mouse.up();rendered(ui)
 for n in bounds(ui):
  if n['type']=='PrimitiveStringMultiline':
   x,y=n['pos'];w,h=n['size'];ui.page.mouse.move(x+w-3,y+h-3);ui.page.mouse.down();ui.page.mouse.move(x+430-3,y+230-3,steps=12);ui.page.mouse.up()
 ui.page.set_viewport_size({'width':2200,'height':1100});rendered(ui)
 b=bounds(ui);pan(ui,100-min(n['pos'][0] for n in b),170-min(n['pos'][1] for n in b));rendered(ui)
 for n in bounds(ui):
  x,y=n['pos'];w,h=n['size'];assert x>=70 and y>=100 and x+w<=2170 and y+h<=1040,(n['type'],x,y,w,h)

with browser_session(args) as ui:
 ui.artifacts=out
 ui.page.on('websocket',lambda ws:ws.on('framereceived',frame))
 ui.page.reload(wait_until='domcontentloaded');ui.page.get_by_test_id('queue-button').wait_for(state='visible')
 ui.app=ui.page.evaluate_handle("async ()=>(await import('/scripts/app.js')).app")
 ui.page.on('request',lambda request:requests.append(request.post_data_json) if request.method=='POST' and urlsplit(request.url).path in ['/prompt','/api/prompt'] else None)
 ui.page.keyboard.press('Escape');close=ui.page.get_by_test_id('close-minimap-button')
 if close.is_visible():close.click()
 ui.new_workflow();ui.load_workflow(path,len(workflow['nodes']))
 full_zoom(ui);show_node(ui,encoder)
 ui.choose('ImpactWildcardEncode','mode','populate');ui.choose('ImpactWildcardEncode','control_after_generate','increment');ui.set_number('ImpactWildcardEncode','seed',731)
 source=ui.add_node('Text (Multiline)','PrimitiveStringMultiline',100,850)
 arrange(ui)
 ui.page.get_by_placeholder('value',exact=True).fill(text)
 ui.connect('PrimitiveStringMultiline','ImpactWildcardEncode',target_slot=input_slot(ui,'ImpactWildcardEncode','wildcard_text'))
 native_roundtrip(ui,encoder);arrange(ui)
 export_workflow(ui,out/'constant-lora-workflow.json')

 def record(label,requested_seed,expected_text,expect_cached):
  before=len(requests);entry=ui.run();images=ui.images(entry);assert len(requests)==before+1
  request=requests[-1];prompt=entry['prompt'][2];inputs=prompt[encoder]['inputs']
  assert request['prompt'][encoder]['inputs']['seed']==requested_seed
  assert request['prompt'][encoder]['inputs']['wildcard_text']==[source,0]
  assert inputs['wildcard_text']==[source,0] and inputs['seed']==0 and inputs['populated_text']==expected_text
  assert prompt[source]['inputs']['value']==expected_text
  assert prompt[sampler]['inputs']['seed']==944516346096519
  observed=[e for e in events if e['data'].get('prompt_id')==entry['prompt'][1]]
  cache_events=[e for e in observed if e['type']=='execution_cached'];assert len(cache_events)==1
  cached_ids=cached(entry);assert cached_ids==set(map(str,cache_events[0]['data']['nodes']))
  executed={str(e['data']['node']) for e in observed if e['type']=='executing' and e['data']['node'] is not None}
  assert (sampler in cached_ids)==expect_cached and (sampler in executed)==(not expect_cached)
  if label=='first':assert not cached_ids
  elif label=='changed':assert cached_ids=={'4','5','7'}
  else:assert cached_ids==set(prompt)
  image=images[output][0];assert image.size==(512,768)
  pixels=hashlib.sha256(image.tobytes()).hexdigest()
  if label=='same':assert pixels==results[0]['pixel_sha256']
  if label=='changed':assert pixels!=results[0]['pixel_sha256']
  if label.startswith('restored'):assert pixels==results[2]['pixel_sha256']
  for suffix,data in [('request',request),('history',entry),('events',observed)]:
   (out/f'{label}-{suffix}.json').write_text(json.dumps(data,indent=2)+'\n')
  image.save(out/f'{label}-output.png')
  arrange(ui);ui.page.mouse.move(2160,1030);ui.page.screenshot(path=str(out/f'{label}.png'))
  (out/f'{label}-state.json').write_text(json.dumps(bounds(ui),indent=2)+'\n')
  results.append({'case':label,'requested_seed':requested_seed,'execution_seed':0,'source_node':source,'source_text':expected_text,'sampler_seed':prompt[sampler]['inputs']['seed'],'cached_nodes':sorted(cached_ids,key=int),'executing_nodes':sorted(executed,key=int),'pixel_sha256':pixels})
  print(json.dumps(results[-1]),flush=True)

 record('first',731,text,False)
 record('same',732,text,True)
 ui.page.get_by_placeholder('value',exact=True).fill(text+edit)
 record('changed',733,text+edit,False)
 for by_name,seed,label in [(False,734,'restored-positional'),(True,735,'restored-named')]:
  setting(ui,'Restore widget values','Comfy.Workflow.NamedValuesRestore',by_name)
  native_roundtrip(ui,encoder);arrange(ui)
  record(label,seed,text+edit,True)
(out/'result.json').write_text(json.dumps({'scope':'Modified LBW tutorial used to exercise direct PrimitiveStringMultiline input, cache reuse, text invalidation and native restoration with real LoRA sampling. Not original-tutorial pixel parity.','adaptations':['Character LoRA replaced with installed SD1.5/etc/add_detail.safetensors at 0.7/0.7; original LBW vector retained','Prompt changed to a cottage landscape','One PrimitiveStringMultiline added through node search and linked through UI','Wildcard populate mode, increment policy and seed 731 set through UI; original checkpoint and KSampler settings retained'],'node_types':{'3':'KSampler','4':'CheckpointLoaderSimple','5':'EmptyLatentImage','7':'CLIPTextEncode (negative)','8':'VAEDecode','9':'SaveImage','10':'ImpactWildcardEncode',source:'PrimitiveStringMultiline'},'cases':results},indent=2)+'\n')
