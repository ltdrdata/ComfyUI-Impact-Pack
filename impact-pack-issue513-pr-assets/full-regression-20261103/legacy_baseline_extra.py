from runlib import *
import hashlib
area=ROOT/'legacy-isolated';pack=area/'runtime/custom_nodes/comfyui-impact-pack';stop=area/'stop.py'
# Keep exactly the same adapted tests and corpus; replace only production sources.
paths=[x for x in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if x.startswith(('modules/','js/')) or x=='__init__.py']
saved={p:(pack/p).read_bytes() for p in paths if (pack/p).is_file()}
changed=[]
try:
 for p,data in saved.items():
  base=subprocess.check_output(['git','show','244e93ef9883aaf241d5d0e54ebb57e3b2e840d3:'+p],cwd=REPO)
  if base!=data:changed.append(p)
  (pack/p).write_bytes(base)
 (ROOT/'legacy-baseline-extra-sources.json').write_text(json.dumps({'base':'244e93ef9883aaf241d5d0e54ebb57e3b2e840d3','source_replacements':len(saved),'different_files':changed},indent=2)+'\n')
 for label,path in [('progressive','tests/wildcards/test_progressive_ondemand.sh'),('quotes','tests/test_config_quotes.sh')]:
  try:run('baseline-legacy-extra-'+label,['bash',pack/path],cwd=pack,timeout=600)
  finally:subprocess.run([PY,stop],check=True)
finally:
 for p,data in saved.items():(pack/p).write_bytes(data)
