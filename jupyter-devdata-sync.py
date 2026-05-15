#!/usr/bin/env python3
import os, shutil, tempfile, time
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_USERNAME = os.environ.get("HF_USERNAME", "").strip() or os.environ.get("SPACE_AUTHOR_NAME", "").strip()
DATASET_NAME = os.environ.get("DEVDATA_DATASET_NAME", "huggingclaw-devdata").strip()
JUPYTER_ROOT = Path(os.environ.get("JUPYTER_ROOT_DIR", "/home/node")).resolve()
INTERVAL = int(os.environ.get("DEVDATA_SYNC_INTERVAL", "300"))
ENABLE = os.environ.get("DEVDATA", "on").strip().lower() not in {"off","false","0","no"}

EXCLUDE = {".cache", "node_modules", ".npm", ".yarn", ".local/share/Trash", ".ipynb_checkpoints"}

def enabled():
    dev = os.environ.get("DEV_MODE", "").strip().lower() in {"1","true","yes","on"}
    return ENABLE and dev and bool(HF_TOKEN)

def repo_id(api: HfApi) -> str:
    ns = HF_USERNAME
    if not ns:
        who = api.whoami()
        ns = who.get("name") or who.get("user") or ""
    if not ns:
        raise RuntimeError("Cannot resolve HF namespace for devdata sync")
    return f"{ns}/{DATASET_NAME}"

def should_skip(p: Path):
    parts = set(p.parts)
    return any(x in parts for x in EXCLUDE)

def snapshot(src: Path, dst: Path):
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if should_skip(rel):
            continue
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, target)
            except OSError:
                pass

def restore_once(api: HfApi, rid: str):
    tmp = Path(tempfile.mkdtemp(prefix="devdata-restore-"))
    try:
        snapshot_download(repo_id=rid, repo_type="dataset", local_dir=str(tmp), local_dir_use_symlinks=False, token=HF_TOKEN)
        for p in tmp.rglob("*"):
            rel = p.relative_to(tmp)
            if should_skip(rel):
                continue
            target = JUPYTER_ROOT / rel
            if p.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif p.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        print(f"DevData restored from {rid}")
    except RepositoryNotFoundError:
        print(f"DevData dataset not found yet: {rid}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def sync_loop(api: HfApi, rid: str):
    while True:
        tmp = Path(tempfile.mkdtemp(prefix="devdata-snap-"))
        try:
            snapshot(JUPYTER_ROOT, tmp)
            upload_folder(folder_path=str(tmp), repo_id=rid, repo_type="dataset", token=HF_TOKEN,
                          commit_message=f"DevData sync {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                          ignore_patterns=[".git/*", ".git"])
            print(f"DevData synced to {rid}")
        except Exception as exc:
            print(f"DevData sync warning: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    if not enabled():
        print("DevData sync disabled.")
        raise SystemExit(0)
    api = HfApi(token=HF_TOKEN)
    rid = repo_id(api)
    try:
        api.repo_info(repo_id=rid, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=rid, repo_type="dataset", private=True)
    restore_once(api, rid)
    sync_loop(api, rid)
