"""Compare recorded requests/events with real ComfyUI cache signatures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import asyncio,hashlib,json,sys
sys.path[:0]=['/mnt/teratera/git/ComfyUI','/mnt/teratera/git/impact-pack-pr1269-followup-pr-assets/full-head-63d28f8/source/modules']
from comfy.cli_args import args
args.cpu=True
from aiohttp import web
from server import PromptServer
PromptServer.instance=SimpleNamespace(routes=web.RouteTableDef(),add_on_prompt_handler=lambda h:None,send_sync=lambda *a:None)
import nodes
# Match the package entrypoint: the route module initializes before impact_pack.
from impact import impact_server, impact_pack
from comfy_extras.nodes_primitive import StringMultiline
from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt
from execution import IsChangedCache
nodes.NODE_CLASS_MAPPINGS.update({'ImpactWildcardEncode':impact_pack.ImpactWildcardEncode,'PrimitiveStringMultiline':StringMultiline})
root=Path(__file__).resolve().parent
rows=json.loads((root/'result.json').read_text())['cases']

def canonical(value):
 if isinstance(value,frozenset):return sorted((canonical(item) for item in value),key=lambda item:json.dumps(item,sort_keys=True))
 if isinstance(value,(tuple,list)):return [canonical(item) for item in value]
 return value

async def main():
 keys={};details={};prompts={}
 for row in rows:
  case=row['case'];history=json.loads((root/f'{case}-history.json').read_text());prompt=history['prompt'][2]
  events=json.loads((root/f'{case}-events.json').read_text())
  assert history['status']['completed'] and history['status']['status_str']=='success'
  assert all(event['data']['prompt_id']==history['prompt'][1] for event in events)
  cache_events=[event['data']['nodes'] for event in events if event['type']=='execution_cached']
  assert len(cache_events)==1 and sorted(cache_events[0],key=int)==row['cached_nodes']
  history_cache=[data['nodes'] for kind,data in history['status']['messages'] if kind=='execution_cached']
  assert history_cache==cache_events
  executed={event['data']['node'] for event in events if event['type']=='executing' and event['data']['node'] is not None}
  assert sorted(executed,key=int)==row['executing_nodes']
  request=json.loads((root/f'{case}-request.json').read_text())['prompt'];normalized=deepcopy(request)
  assert request['10']['inputs']['seed']==row['requested_seed']
  normalized['10']['inputs'].update(seed=0,mode='reproduce',populated_text=request[row['source_node']]['inputs']['value'])
  assert normalized==prompt
  assert prompt['3']['inputs']['seed']==944516346096519
  dynamic=DynamicPrompt(prompt);builder=CacheKeySetInputSignature(dynamic,prompt.keys(),IsChangedCache(case,dynamic,None))
  await builder.add_keys(prompt.keys());keys[case]={node:builder.get_data_key(node) for node in prompt}
  prompts[case]=prompt
  details[case]={node:{'class_type':value['class_type'],'inputs':value['inputs'],'ancestors':builder.get_ordered_ancestry(dynamic,node)[0],'signature_sha256':hashlib.sha256(json.dumps(canonical(keys[case][node]),sort_keys=True,allow_nan=False).encode()).hexdigest()} for node,value in prompt.items()}
 comparisons=[]
 for previous,current in zip(rows,rows[1:]):
  before=previous['case'];after=current['case']
  same=sorted([n for n in keys[after] if keys[before][n]==keys[after][n]],key=int)
  changed=sorted([n for n in keys[after] if n not in same],key=int)
  assert same==current['cached_nodes'];assert changed==current['executing_nodes']
  differences=[{'node':n,'class_type':prompts[after][n]['class_type'],'input':k,'before':prompts[before][n]['inputs'][k],'after':v} for n in prompts[after] for k,v in prompts[after][n]['inputs'].items() if prompts[before][n]['inputs'][k]!=v]
  comparisons.append({'previous':before,'current':after,'input_changes':differences,'same_cache_keys':same,'changed_cache_keys':changed})
 assert comparisons[1]['same_cache_keys']==['4','5','7']
 assert {(d['node'],d['input']) for d in comparisons[1]['input_changes']}=={('16','value'),('10','populated_text')}
 (root/'node-cache-keys.json').write_text(json.dumps({'method':'Real CacheKeySetInputSignature, independently recomputed from captured execution prompts; compared with actual cached/executing events','cases':details,'comparisons':comparisons},indent=2)+'\n')
 print('PASS: four transitions match captured cache/execution events and real cache signatures.')
asyncio.run(main())
