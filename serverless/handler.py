"""Handler Runpod Serverless : ComfyUI (Krea 2) -> Google Drive via rclone.

Entrée  : {"input": {"prompt": "...", "workflow": "KREA2", "negative": "...", "seed": 123, "count": 1,
                     "folder": "2026-10/2026-10-18", "overrides": {"11.megapixel": "2.4"}}}
Sortie  : {"status": "ok", "workflow", "files": [{"file", "drive_path", "seed"}], "drive_folder", "seconds"}
Seul "prompt" est obligatoire. "workflow" = nom du fichier dans Workflows/api/ (avec ou sans .json),
KREA2.json par défaut. Nœuds repérés par leur titre ComfyUI : API_PROMPT, API_NEGATIVE, API_SEED, API_SAVE ;
à défaut, ids 9 / 10 / 7 / 21 (KREA2.json n'a pas besoin de titres).
"""
import base64
import filecmp
import json
import os
import random
import shlex
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import runpod

# ── Config (surchargeable via les variables d'env de l'endpoint) ──────────────
VOLUME = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/workspace"
COMFY_DIR = os.environ.get("COMFY_DIR", f"{VOLUME}/ComfyUI")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
COMFY_LOG = "/tmp/comfyui.log"
OUT_DIR = "/tmp/comfy_out"        # sorties ComfyUI hors volume
USER_DIR = "/tmp/comfy_user"      # user dir isolé : le volume (partagé avec le Pod) n'est pas modifié
BOOT_TIMEOUT = int(os.environ.get("COMFY_BOOT_TIMEOUT", "420"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

GITHUB_REPO = os.environ.get("GITHUB_REPO", "Bigornh2o/workflow_bigornh2o")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "KREA2.json")  # workflow par défaut
WORKFLOW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/Workflows/api/"
WORKFLOW_CACHE_DIR = "/tmp/workflow_cache"

NODE_PROMPT = os.environ.get("NODE_PROMPT", "9")
NODE_NEGATIVE = os.environ.get("NODE_NEGATIVE", "10")
NODE_SEED = os.environ.get("NODE_SEED", "7")
NODE_SAVE = os.environ.get("NODE_SAVE", "21")

DRIVE_REMOTE = os.environ.get("DRIVE_REMOTE", "gdrive")
DRIVE_ROOT_ID = os.environ.get("DRIVE_ROOT_FOLDER_ID", "1lqzuiyED9wwkSsS8ayjLGqa7ieGskr8r")
RCLONE_CONF = "/root/.config/rclone/rclone.conf"
TZ = ZoneInfo("Europe/Paris")

CLIENT_ID = str(uuid.uuid4())
_lock = threading.Lock()
_proc = None


def log(*a):
    print("[handler]", *a, flush=True)


# ── ComfyUI ───────────────────────────────────────────────────────────────────
def comfy_up():
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def log_tail(n=40):
    try:
        with open(COMFY_LOG, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(pas de log)"


def prepare_volume():
    # Nœud Flux Resolution Calc patché (copié à la racine de l'image, comme au boot du Pod)
    patched = "/flux_resolution_cal_node.py"
    dst = f"{COMFY_DIR}/custom_nodes/controlaltai-nodes/flux_resolution_cal_node.py"
    if os.path.isfile(patched) and os.path.isdir(os.path.dirname(dst)):
        if not (os.path.isfile(dst) and filecmp.cmp(patched, dst, shallow=False)):
            shutil.copyfile(patched, dst)
            log("Flux Resolution Calc patché")
    if not os.path.isdir(f"{COMFY_DIR}/custom_nodes/ResolutionScaler"):
        log("ATTENTION : custom node ResolutionScaler absent du volume (lance le Pod une fois pour le cloner)")
    # ComfyUI-Manager hors ligne (config dans le user dir isolé)
    cfg = f"{USER_DIR}/default/ComfyUI-Manager"
    os.makedirs(cfg, exist_ok=True)
    with open(f"{cfg}/config.ini", "w") as f:
        f.write("[default]\nnetwork_mode = offline\nsecurity_level = normal\n")


def boot():
    """Démarre ComfyUI si besoin et attend qu'il réponde."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None and comfy_up():
            return
        if not os.path.isfile(f"{COMFY_DIR}/main.py"):
            raise RuntimeError(f"ComfyUI introuvable dans {COMFY_DIR} : volume réseau non monté sur l'endpoint ?")
        prepare_volume()
        os.makedirs(OUT_DIR, exist_ok=True)

        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)  # comme `env -u LD_PRELOAD` du start script
        env.update(
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
            TORCH_HOME=f"{VOLUME}/.torch_cache",
            TRITON_CACHE_DIR=f"{VOLUME}/.triton_cache",  # cache Triton persistant sur le volume
        )
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in [
                "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64",
                "/opt/venv/lib/python3.12/site-packages/torch/lib", env.get("LD_LIBRARY_PATH"),
            ] if p
        )
        cmd = [
            "/opt/venv/bin/python", "-u", f"{COMFY_DIR}/main.py",
            "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
            "--use-sage-attention", "--disable-pinned-memory", "--cache-ram", "4",
            "--output-directory", OUT_DIR, "--user-directory", USER_DIR,
            "--disable-auto-launch",
        ] + shlex.split(os.environ.get("COMFY_EXTRA_ARGS", ""))

        log("Démarrage ComfyUI :", " ".join(cmd))
        _proc = subprocess.Popen(cmd, cwd=COMFY_DIR, env=env,
                                 stdout=open(COMFY_LOG, "ab"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < BOOT_TIMEOUT:
            if _proc.poll() is not None:
                raise RuntimeError(f"ComfyUI s'est arrêté (code {_proc.returncode}) :\n{log_tail()}")
            if comfy_up():
                log(f"ComfyUI prêt en {time.time() - t0:.0f}s")
                return
            time.sleep(1)
        raise RuntimeError(f"ComfyUI n'a pas répondu en {BOOT_TIMEOUT}s :\n{log_tail()}")


def comfy_get(path):
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=30) as r:
        return json.load(r)


def comfy_post(path, payload):
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI a refusé le workflow ({e.code}) : {e.read().decode(errors='replace')[:1500]}")


# ── Workflow ──────────────────────────────────────────────────────────────────
def clean_workflow_name(name):
    name = (name or WORKFLOW_NAME).strip()
    if not name.lower().endswith(".json"):
        name += ".json"
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("input.workflow invalide : nom de fichier seul, sans dossier")
    return name


def fetch_workflow(name=None):
    """Télécharge Workflows/api/<name> depuis GitHub raw à chaque job (copie locale en secours)."""
    name = clean_workflow_name(name)
    os.makedirs(WORKFLOW_CACHE_DIR, exist_ok=True)
    cache = os.path.join(WORKFLOW_CACHE_DIR, name)
    try:
        req = urllib.request.Request(WORKFLOW_BASE + urllib.parse.quote(name),
                                     headers={"User-Agent": "runpod-comfy-handler"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        wf = json.loads(data)
        with open(cache, "wb") as f:
            f.write(data)
        return wf
    except Exception as e:
        if os.path.isfile(cache):
            log(f"GitHub injoignable ({e}), copie locale utilisée")
            with open(cache) as f:
                return json.load(f)
        raise RuntimeError(f"Workflow '{name}' introuvable dans Workflows/api/ : {e}")


def find_node(wf, role, default_id):
    """Nœud dont le titre ComfyUI est `role` (ex. API_PROMPT), sinon l'id par défaut."""
    for nid, node in wf.items():
        if isinstance(node, dict) and str((node.get("_meta") or {}).get("title", "")).strip().upper() == role:
            return nid
    return str(default_id)


def node_inputs(wf, node_id):
    node = wf.get(str(node_id))
    if not isinstance(node, dict) or "class_type" not in node:
        raise ValueError(f"nœud {node_id} absent du JSON (export au format API ?)")
    return node.setdefault("inputs", {})


def set_text(wf, node_id, text):
    inputs = node_inputs(wf, node_id)
    key = next((k for k in ("text", "prompt", "string", "value") if isinstance(inputs.get(k), str)), None)
    if key is None:
        raise ValueError(f"nœud {node_id} : pas d'entrée texte directe (branchée sur un autre nœud ?)")
    inputs[key] = text


def run_workflow(wf):
    pid = comfy_post("/prompt", {"prompt": wf, "client_id": CLIENT_ID})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < JOB_TIMEOUT:
        if _proc.poll() is not None:
            raise RuntimeError(f"ComfyUI a planté pendant la génération :\n{log_tail()}")
        entry = comfy_get(f"/history/{pid}").get(pid)
        if entry:
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError("Erreur ComfyUI : " + json.dumps(st.get("messages"))[-1500:])
            if st.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(1)
    raise TimeoutError(f"génération > {JOB_TIMEOUT}s")


def output_files(entry, save_id):
    outs = entry.get("outputs", {})
    imgs = (outs.get(str(save_id)) or {}).get("images") or [
        i for o in outs.values() for i in o.get("images", []) if i.get("type") == "output"
    ]
    return [os.path.join(OUT_DIR, i.get("subfolder", ""), i["filename"]) for i in imgs]


# ── Google Drive (rclone) ─────────────────────────────────────────────────────
def setup_rclone():
    if os.path.isfile(RCLONE_CONF):
        return
    b64 = os.environ.get("RCLONE_CONF_B64")
    if not b64:
        raise RuntimeError("RCLONE_CONF_B64 absent : secret Runpod non branché sur l'endpoint")
    os.makedirs(os.path.dirname(RCLONE_CONF), exist_ok=True)
    with open(RCLONE_CONF, "wb") as f:
        f.write(base64.b64decode(b64))
    os.chmod(RCLONE_CONF, 0o600)


def upload(path, dest_dir):
    name = os.path.basename(path)
    r = subprocess.run(
        ["rclone", "copyto", path, f"{DRIVE_REMOTE}:{dest_dir}/{name}",
         "--config", RCLONE_CONF, "--drive-root-folder-id", DRIVE_ROOT_ID],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode:
        raise RuntimeError(f"rclone a échoué : {r.stderr[-800:]}")
    return f"{dest_dir}/{name}"


# ── Job ───────────────────────────────────────────────────────────────────────
def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "input.prompt manquant"}
    try:
        wf_name = clean_workflow_name(inp.get("workflow"))
        count = max(1, min(int(inp.get("count", 1)), 8))
        seed0 = int(inp["seed"]) if inp.get("seed") is not None else random.randint(0, 2**32 - 1)
        now = datetime.now(TZ)
        dest = (inp.get("folder") or f"{now:%Y-%m}/{now:%Y-%m-%d}").strip("/")
        if ".." in dest.split("/"):
            return {"error": "input.folder invalide"}

        setup_rclone()  # échoue avant de consommer du GPU si le secret manque
        boot()

        results = []
        for i in range(count):
            wf = fetch_workflow(wf_name)
            seed = seed0 + i
            set_text(wf, find_node(wf, "API_PROMPT", NODE_PROMPT), prompt)
            if inp.get("negative") is not None:
                set_text(wf, find_node(wf, "API_NEGATIVE", NODE_NEGATIVE), str(inp["negative"]))
            node_inputs(wf, find_node(wf, "API_SEED", NODE_SEED))["seed"] = seed
            for key, val in (inp.get("overrides") or {}).items():  # "11.megapixel": "2.4"
                nid, name = key.split(".", 1)
                node_inputs(wf, nid)[name] = val

            for path in output_files(run_workflow(wf), find_node(wf, "API_SAVE", NODE_SAVE)):
                results.append({"file": os.path.basename(path), "drive_path": upload(path, dest), "seed": seed})
                os.remove(path)

        if not results:
            return {"error": "aucune image produite par le workflow"}
        return {"status": "ok", "workflow": wf_name, "files": results, "drive_folder": dest,
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        log(traceback.format_exc())
        return {"error": f"{type(e).__name__}: {e}"}


def _boot_bg():
    try:
        boot()
    except Exception as e:
        log("boot en arrière-plan échoué :", e)


# ComfyUI démarre pendant l'init du worker ; le premier job attend sur le verrou si besoin
threading.Thread(target=_boot_bg, daemon=True).start()
runpod.serverless.start({"handler": handler})
