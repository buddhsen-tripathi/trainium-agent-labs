from pathlib import Path
import subprocess,sys,os,json
root=Path(__file__).resolve().parent
os.chdir(root)
env=os.environ.copy()
env.pop("NEURON_RT_VISIBLE_CORES",None)
env.update(NEURON_RT_NUM_CORES="1",NEURON_LOGICAL_NC_CONFIG="2",NEURON_PLATFORM_TARGET_OVERRIDE="trn2",PYTHONDONTWRITEBYTECODE="1")
cmd=[sys.executable,"-u","-B","agent.py","--level","8","--rounds","3","--samples","2","--repeat","2","--context","8192","--max-tokens","3000","--base","http://localhost:8000/v1","--model","Qwen/Qwen3-8B","--lint","--level-hints","--portfolio","--echo-check","--log","results/agent-fresh.jsonl"]
with (root/"results/agent-fresh.log").open("w") as log:
 result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
report={"agent_returncode":result.returncode,"command":cmd,"runs":[]}
rows=[json.loads(l) for l in (root/"results/agent-fresh.jsonl").read_text().splitlines() if l.strip()]
for run in range(2):
 good=[r for r in rows if r.get("run")==run and r.get("parts",{}).get("correct") and r.get("selected")]
 if not good:
  report["runs"].append({"run":run,"solved":False}); continue
 record=good[0]
 candidate="solved/level08_generated_run"+str(run)+".py"
 (root/candidate).write_text(record["code"]+"\n")
 item={"run":run,"solved":True,"round":record["round"],"sample":record["sample"],"candidate":candidate,"checks":{}}
 for name,check in [
  ("independent",[sys.executable,"-B","tests/check_attention.py",candidate,"--json","results/generated-run"+str(run)+"-independent.json"]),
  ("device",[sys.executable,"-u","-B","tests/check_attention_device.py",candidate,"--lnc","2","--suite","official","--json","results/generated-run"+str(run)+"-device.json","--stop-on-error"])]:
  print("START",run,name,flush=True)
  with (root/"results"/("generated-run"+str(run)+"-"+name+".log")).open("w") as log:
   done=subprocess.run(check,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  item["checks"][name]=done.returncode
  print("DONE",run,name,done.returncode,flush=True)
 report["runs"].append(item)
 (root/"results/generation-summary.json").write_text(json.dumps(report,indent=2)+"\n")
(root/"results/generation-summary.json").write_text(json.dumps(report,indent=2)+"\n")
print("GENERATION_COMPLETE",flush=True)
