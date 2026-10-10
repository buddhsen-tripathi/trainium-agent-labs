from pathlib import Path
import os,sys,json,traceback,hashlib
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]="trn2"
root=Path("/workspace/level8-repair-shubham-97-planned")
sys.path.insert(0,str(root))
import nkibench as nb
from nki.framework.compiled import CompileKernel,compile_kernel_to_nir
from nki.compiler.driver import _compile_bir_to_neff
fn=nb.load_kernel(str(root/"solved/level08_attention.py"),"nki_attention_")
records=[]
for shape in nb.LEVELS[8]["shapes"]:
 name=nb.label(shape,8)
 print("COMPILE_ONLY_START",name,flush=True)
 row={"case":name,"compiled":False,"device_executed":False}
 try:
  folder=root/"results/device-compile"/f"seq{shape['seq']}-dim{shape['dim']}"
  folder.mkdir(parents=True,exist_ok=True)
  k=fn[2]._to_subclass(CompileKernel,target="trn2",artifacts_dir=str(folder))
  args,_=nb.make_inputs(shape,8,seed=0)
  inputs=k._convert_inputs(k._bind_args(args,{}))
  opts=k._compile_opts()
  nir=compile_kernel_to_nir(k.func,inputs=inputs,compile_opts=opts)
  compiled=_compile_bir_to_neff(nir,opts,inputs)
  path=Path(compiled.neff_path)
  row.update(compiled=True,neff=str(path),neff_bytes=path.stat().st_size,neff_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 except Exception as e:
  row.update(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
  print(row["traceback"],flush=True)
 records.append(row)
 (root/"results/device-compile-only.json").write_text(json.dumps(records,indent=2)+"\n")
 print(json.dumps(row),flush=True)
print("COMPILE_ONLY_COMPLETE",sum(r["compiled"] for r in records),"/",len(records),flush=True)
raise SystemExit(0 if all(r["compiled"] for r in records) else 1)
