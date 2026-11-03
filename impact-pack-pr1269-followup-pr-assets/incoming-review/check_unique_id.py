"""Compare real core cache keys using Main and follow-up wildcard declarations."""
import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

repo = Path('/mnt/teratera/git/impact-pack-pr1269-followup')
core = Path('/mnt/teratera/git/ComfyUI')
sys.path[:0] = [str(core), str(repo / 'modules')]
from comfy.cli_args import args
args.cpu = True
from aiohttp import web
from server import PromptServer
PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
                                        send_sync=lambda *args: None)
import nodes
from comfy_execution import caching
from comfy_execution.graph import DynamicPrompt
from comfy_extras.nodes_string import StringConcatenate
from execution import IsChangedCache
from impact import impact_server, impact_pack

baseline = json.loads((Path(__file__).resolve().parents[1] / 'source.json').read_text())['main']
names = ('ImpactWildcardProcessor', 'ImpactWildcardEncode')
source = subprocess.check_output(['git', 'show', baseline + ':modules/impact/impact_pack.py'], cwd=repo, text=True)
tree = ast.parse(source)
namespace = vars(impact_pack).copy()
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
assert len(classes) == 2
exec(compile(ast.Module(body=classes, type_ignores=[]), baseline + ':impact_pack.py', 'exec'), namespace)
nodes.NODE_CLASS_MAPPINGS['StringConcatenate'] = StringConcatenate
observations = []


async def inspect(label, name, cls):
    nodes.NODE_CLASS_MAPPINGS[name] = cls
    caching.NODE_CLASS_CONTAINS_UNIQUE_ID.clear()
    inputs = {'wildcard_text': '', 'populated_text': 'plain', 'mode': 'fixed', 'seed': 0}
    graph = {}
    slot = 0
    if name == 'ImpactWildcardEncode':
        # No inference: this shared upstream input only supplies a cache signature.
        graph['checkpoint'] = {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'shared-checkpoint'}}
        inputs.update(model=['checkpoint', 0], clip=['checkpoint', 1])
        slot = 3
    graph.update({key: {'class_type': name, 'inputs': deepcopy(inputs)} for key in ('a', 'b')})
    graph.update({f'out_{key}': {'class_type': 'StringConcatenate',
                  'inputs': {'string_a': [key, slot], 'string_b': '', 'delimiter': ''}} for key in ('a', 'b')})
    dynamic = DynamicPrompt(graph)
    keys = caching.CacheKeySetInputSignature(dynamic, graph.keys(), IsChangedCache('review', dynamic, None))
    await keys.add_keys(graph.keys())
    result = {'revision': label, 'node_type': name,
              'hidden_unique_id': caching.include_unique_id_in_input(name),
              'consumer_has_unique_id': caching.include_unique_id_in_input('StringConcatenate'),
              'wildcard_siblings_share_key': keys.get_data_key('a') == keys.get_data_key('b'),
              'downstream_siblings_share_key': keys.get_data_key('out_a') == keys.get_data_key('out_b')}
    assert result['consumer_has_unique_id'] is False
    expected_shared = label == 'Main before #1269'
    assert result['wildcard_siblings_share_key'] == expected_shared
    assert result['downstream_siblings_share_key'] == expected_shared
    observations.append(result)


async def main():
    for name in names:
        await inspect('Main before #1269', name, namespace[name])
        await inspect('432fd24 including #1269', name, getattr(impact_pack, name))


asyncio.run(main())
report = {'scope': 'Actual ComfyUI cache-key computation; no node execution, model loading or timing measurement. '
                   'Baseline declarations are extracted unchanged from Main; current declarations are imported normally.',
          'observations': observations}
Path(__file__).with_name('unique-id-cache-keys.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
