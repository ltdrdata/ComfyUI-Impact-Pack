from argparse import Namespace
from pathlib import Path
import json,sys
source=Path(sys.argv[1]);url=sys.argv[2];artifacts=Path(sys.argv[3]);artifacts.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(source/'tests'))
from e2e_ui import browser_session
from wildcard_tutorial_cases import tutorial
path,workflow,target=tutorial('/mnt/teratera/git/ComfyUI-extension-tutorials','lbw')
args=Namespace(base_url=url,browser_channel='chrome',headed=False,timeout=900,artifacts=artifacts)
with browser_session(args) as ui:
 ui.page.keyboard.press('Escape');ui.new_workflow();ui.load_workflow(path,len(workflow['nodes']))
 entry=ui.run();images=ui.images(entry)
 assert len(images['9'])==1
 images['9'][0].save(artifacts/'reference.png')
 (artifacts/'history.json').write_text(json.dumps(entry,indent=2)+'\n')
 print('PASS: original LBW workflow executed through Chrome on baseline; reference saved')
