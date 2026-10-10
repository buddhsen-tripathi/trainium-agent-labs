"""Run the original NumPy challenge using a Trainium-hosted model.

The unchanged kernelbench.py supplies every reference, input, rule, and tolerance.
Model inference uses Trainium; candidate verification uses the CPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import kernelbench as kb


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluate(candidate, level, output):
    source = candidate.read_text()
    result = dict(level=level, total=len(kb.LEVELS[level]["build"]()), passed=0,
                  failures=[], ok=False, violations=kb.check_rules(source, level))
    try:
        module_spec = importlib.util.spec_from_file_location("candidate", candidate)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        result.update(kb.verify(module.kernel, level, tol=1e-4, stop_early=False))
        result["numerically_correct"] = result["ok"]
        result["ok"] = result["ok"] and not result["violations"]
    except Exception:
        result["error"] = traceback.format_exc()
    save(output, result)


def extract(reply):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", reply or "", re.S)
    if blocks:
        return max(blocks, key=len).strip()
    lines = (reply or "").splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(import |from |def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


def prompt(level, previous=None):
    definition = kb.LEVELS[level]
    reference = inspect.getsource(definition["ref"])
    reference = reference.replace(f"def {definition['ref'].__name__}(", "def kernel(", 1)
    task = (f"Write a NumPy function named kernel matching this reference and its signature:\n"
            f"{reference}\n"
            "Use explicit loops over tiles of at most 128 rows and 512 columns. "
            "Handle partial tiles and dimensions of one. Return one complete Python code block, "
            "including imports, with no prose.\n")
    if previous:
        source, verdict = previous
        feedback = dict(verdict)
        feedback["failures"] = feedback.get("failures", [])[:3]
        task += (f"Repair this previous attempt:\n```python\n{source}\n```\n"
                 f"Original checker feedback:\n{json.dumps(feedback)}\n"
                 "Replace banned whole-operation calls with explicit tiled arithmetic and loops. "
                 "Preserve the reference calculation and do not loosen its tolerance.\n")
    return task


def run(options):
    import httpx
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = output / "candidates"
    candidates.mkdir(exist_ok=True)
    source = Path(kb.__file__)
    client = httpx.Client(timeout=240)
    models = client.get(options.base.rstrip("/") + "/models")
    models.raise_for_status()
    hardware = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True)
    selftest = subprocess.run([sys.executable, "-B", str(source), "--selftest"],
                              capture_output=True, text=True)
    (output / "harness-selftest.log").write_text(selftest.stdout + selftest.stderr)
    if selftest.returncode:
        raise RuntimeError("Original harness selftest failed")
    report = dict(suite="original repository kernelbench.py", host=os.uname().nodename,
                  model=options.model, model_endpoint=options.base,
                  model_hardware="Trainium", verification_hardware="CPU (NumPy)",
                  harness_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tolerance=1e-4, supplemental_cases_added=False,
                  rounds=options.rounds, samples=options.samples, max_tokens=options.max_tokens,
                  context=8192, thinking=False, models=models.json(),
                  hardware=json.loads(hardware.stdout), levels=[], status="running")
    save(output / "summary.json", report)

    def ask(text, sample):
        temperature = (0.2, 0.8)[sample % 2]
        started = time.monotonic()
        response = client.post(options.base.rstrip("/") + "/chat/completions", json=dict(
            model=options.model, messages=[dict(role="user", content=text)],
            max_tokens=min(options.max_tokens, max(256, 8192 - len(text)//3 - 128)),
            temperature=temperature, top_p=0.95,
            chat_template_kwargs=dict(enable_thinking=False)))
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        return dict(source=extract(choice["message"].get("content")), response=body,
                    request_wall_s=time.monotonic()-started, temperature=temperature,
                    finish_reason=choice["finish_reason"])

    with (output / "attempts.jsonl").open("a", buffering=1) as log:
        for level in sorted(kb.LEVELS):
            best, previous = None, None
            print(f"LEVEL_START {level} {kb.LEVELS[level]['name']}", flush=True)
            for round_index in range(options.rounds):
                text = prompt(level, previous)
                round_best = None
                with ThreadPoolExecutor(max_workers=options.samples) as pool:
                    jobs = [pool.submit(ask, text, sample) for sample in range(options.samples)]
                    for sample, job in enumerate(jobs):
                        stem = f"level{level:02d}-round{round_index:02d}-sample{sample}"
                        path = candidates / (stem + ".py")
                        verdict_path = candidates / (stem + ".json")
                        record = dict(level=level, round=round_index, sample=sample, prompt=text)
                        try:
                            record.update(job.result())
                            path.write_text(record["source"] + "\n")
                            try:
                                checked = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--evaluate", str(path), "--level", str(level),
                                    "--output", str(verdict_path)], capture_output=True, text=True, timeout=45)
                                verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else dict(
                                    ok=False, passed=0, error=checked.stderr[-2000:])
                            except subprocess.TimeoutExpired:
                                verdict = dict(ok=False, passed=0, error="candidate evaluation exceeded 45s")
                            record["verdict"] = verdict
                            key = (bool(verdict["ok"]), not bool(verdict.get("violations")),
                                   verdict["passed"], -len(verdict.get("violations", [])))
                            if best is None or key > best[0]:
                                best = (key, record["source"], verdict, stem)
                            if round_best is None or key > round_best[0]:
                                round_best = (key, record["source"], verdict, stem)
                        except Exception:
                            record["error"] = traceback.format_exc()
                        log.write(json.dumps(record) + "\n")
                        print(json.dumps({k: record[k] for k in ("level", "round", "sample", "verdict", "error")
                                          if k in record}), flush=True)
                if best:
                    if round_best:
                        previous = round_best[1:3]
                    if best[2]["ok"]:
                        break
            if best:
                selected = output / f"level{level:02d}.py"
                selected.write_text(best[1] + "\n")
                entry = dict(level=level, name=kb.LEVELS[level]["name"], candidate=str(selected),
                             selected_attempt=best[3], sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                             **{k:v for k,v in best[2].items() if k != "level"})
            else:
                entry = dict(level=level, name=kb.LEVELS[level]["name"], ok=False, passed=0,
                             total=len(kb.LEVELS[level]["build"]()), error="No candidate returned")
            report["levels"].append(entry)
            save(output / "summary.json", report)
            print("LEVEL_DONE " + json.dumps(entry), flush=True)
    report.update(status="complete", levels_passed=sum(x["ok"] for x in report["levels"]),
                  cases_passed=sum(x["passed"] for x in report["levels"]),
                  total_cases=sum(x["total"] for x in report["levels"]))
    save(output / "summary.json", report)
    print(f"COMPLETE {report['levels_passed']}/10 levels, {report['cases_passed']}/{report['total_cases']} numerical cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--level", type=int)
    arguments = parser.parse_args()
    if arguments.evaluate:
        evaluate(arguments.evaluate, arguments.level, arguments.output)
    else:
        run(arguments)
