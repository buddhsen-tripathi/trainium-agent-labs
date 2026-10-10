from pathlib import Path
import subprocess,sys,os,json,time,hashlib
root=Path(__file__).resolve().parent
os.chdir(root)
env=os.environ.copy()
env.pop("NEURON_RT_VISIBLE_CORES",None)
env.update(NEURON_RT_NUM_CORES="1",NEURON_LOGICAL_NC_CONFIG="2",NEURON_PLATFORM_TARGET_OVERRIDE="trn2",PYTHONDONTWRITEBYTECODE="1")
jobs=[
("original-official",[sys.executable,"-B","nkibench.py","--level","8","--check","original/original_level08_attention.py"]),
("adapted-official",[sys.executable,"-B","nkibench.py","--level","8","--check","solved/level08_seat95_adapted.py"]),
("adapted-independent",[sys.executable,"-B","tests/check_attention.py","solved/level08_seat95_adapted.py","--json","results/adapted-independent.json"]),
("adapted-device",[sys.executable,"-u","-B","tests/check_attention_device.py","solved/level08_seat95_adapted.py","--lnc","2","--suite","official","--json","results/adapted-device.json","--stop-on-error"])]
summary={}
for name,cmd in jobs:
 print("START",name,flush=True)
 with (root/"results"/(name+".log")).open("w") as log:
  done=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 summary[name]={"returncode":done.returncode,"command":cmd}
 (root/"results/validation-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print("DONE",name,done.returncode,flush=True)
 if name!="original-official" and done.returncode: sys.exit(done.returncode)
print("VALIDATION_COMPLETE",flush=True)
