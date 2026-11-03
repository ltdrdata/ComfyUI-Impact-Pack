"""Native Chrome reproduction of mixed wildcard/static list cache behavior."""
from argparse import Namespace
from pathlib import Path
from urllib.parse import urlsplit
import json,os,sys
sys.path.insert(0,'/mnt/teratera/git/impact-pack-pr1269-followup-pr-assets/full-head-63d28f8/source/tests')
from e2e_ui import browser_session
from test_e2e_widget_catalog import full_zoom,show_node,setting,export_workflow,pan,rendered
from test_e2e_wildcard_cache import graph,input_slot,native_roundtrip,set_text,cached
out=Path(os.environ['IMPACT_E2E_ARTIFACTS']);out.mkdir(parents=True,exist_ok=True)
args=Namespace(base_url=os.environ['IMPACT_E2E_BASE_URL'],browser_channel='chrome',headed=False,timeout=120,artifacts=out)
requests=[];events=[];observations=[]
def arrange(ui):
    ui.page.set_viewport_size({'width':3000,'height':1800});full_zoom(ui)
    def bounds():
        return ui.app.evaluate('app=>app.graph._nodes.map(n=>({id:String(n.id),type:n.type,pos:app.canvasPosToClientPos(n.pos),size:Array.from(n.size)}))')
    b=bounds();pan(ui,100-min(n['pos'][0] for n in b),180-min(n['pos'][1] for n in b))
    positions={'PrimitiveString':(100,180),'PrimitiveStringMultiline':(100,480),'ImpactMakeAnyList':(580,370),'ImpactWildcardProcessor':(890,180),'PreviewAny':(1390,220)}
    for n in bounds():
        x,y=n['pos'];tx,ty=positions[n['type']]
        ui.page.mouse.move(x+65,y-15);ui.page.mouse.down();ui.page.mouse.move(tx+65,ty-15,steps=12);ui.page.mouse.up();rendered(ui)
    ui.page.set_viewport_size({'width':1800,'height':900});rendered(ui)
    b=bounds();pan(ui,100-min(n['pos'][0] for n in b),180-min(n['pos'][1] for n in b));rendered(ui)
    close=ui.page.get_by_test_id('close-minimap-button')
    if close.is_visible():close.click()
    for n in bounds():
        x,y=n['pos'];w,h=n['size'];assert x>=70 and y>=100 and x+w<1770 and y+h<840,(n,x,y,w,h)
    return bounds()
with browser_session(args) as ui:
    ui.artifacts=out
    ui.page.on('websocket',lambda ws:ws.on('framereceived',lambda payload:events.append(json.loads(payload)) if isinstance(payload,str) else None))
    ui.page.reload(wait_until='domcontentloaded');ui.page.get_by_test_id('queue-button').wait_for(state='visible')
    ui.app=ui.page.evaluate_handle("async ()=>(await import('/scripts/app.js')).app")
    ui.page.on('request',lambda req:requests.append(req.post_data_json) if req.method=='POST' and urlsplit(req.url).path in ['/prompt','/api/prompt'] else None)
    ui.page.keyboard.press('Escape');ui.new_workflow();full_zoom(ui)
    ui.page.set_viewport_size({'width':1800,'height':1200})
    node,preview=graph(ui,'ImpactWildcardProcessor')
    ui.choose('ImpactWildcardProcessor','mode','fixed')
    text1=ui.add_node('Text','PrimitiveString',100,650)
    set_text(ui,'PrimitiveString','value','{red|green|blue}')
    text2=ui.add_node('Text (Multiline)','PrimitiveStringMultiline',100,850)
    ui.page.get_by_placeholder('value',exact=True).fill('plain')
    source=ui.add_node('Make List (Any)','ImpactMakeAnyList',700,800)
    ui.connect('PrimitiveString','ImpactMakeAnyList',target_slot=input_slot(ui,'ImpactMakeAnyList','value1'))
    ui.connect('PrimitiveStringMultiline','ImpactMakeAnyList',target_slot=input_slot(ui,'ImpactMakeAnyList','value2'))
    ui.connect('ImpactMakeAnyList','ImpactWildcardProcessor',target_slot=input_slot(ui,'ImpactWildcardProcessor','populated_text'))
    for named,seed,expected in [(False,2,['red','plain']),(True,5,['blue','plain'])]:
        setting(ui,'Restore widget values','Comfy.Workflow.NamedValuesRestore',named)
        show_node(ui,node);ui.set_number('ImpactWildcardProcessor','seed',seed)
        native_roundtrip(ui,node,source)
        start=len(requests);entry=ui.run();assert len(requests)==start+1
        request=requests[-1];assert request['prompt'][node]['inputs']['seed']==seed
        observed=[e for e in events if e.get('data',{}).get('prompt_id')==entry['prompt'][1]]
        actual=entry['outputs'][preview]['text']
        row={'requested_seed':seed,'execution_seed':entry['prompt'][2][node]['inputs']['seed'],'expected':expected,'actual':actual,'matches_expected':actual==expected,'cached_nodes':sorted(cached(entry),key=int),'wildcard_node':node,'preview_node':preview}
        observations.append(row)
        for label,data in [('request',request),('history',entry),('events',observed)]:
            (out/f'seed-{seed}-{label}.json').write_text(json.dumps(data,indent=2)+'\n')
        geometry=arrange(ui);(out/f'seed-{seed}-layout.json').write_text(json.dumps(geometry,indent=2)+'\n')
        ui.page.mouse.move(1760,850);ui.page.screenshot(path=str(out/f'seed-{seed}.png'))
        print(json.dumps(row),flush=True)
    export_workflow(ui,out/'mixed-list-workflow.json')
(out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
# Record the wrong output before reporting the regression to the runner.
assert all(o['matches_expected'] for o in observations),observations
