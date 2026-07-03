import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/theAbyssOfTime2004/thesis-ttt-sdpo /root/repo",
        "pip install torch",
        "pip install -r /root/repo/requirements.txt",
        "pip install -U transformers peft",   # Gemma4 + all-linear (khớp Colab)
        "pip install trl==1.6.0",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ttt-outputs", create_if_missing=True)

app = modal.App("ttt-sdpo")

@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=32768,
    timeout=28800,
    retries=2,  # auto-restart if Modal preempts/kills the container (SIGINT mid-run)
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/repo/experiments/ttt_trl/outputs": out_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("zai"),
        modal.Secret.from_name("huggingface"),
    ],
)
def run(cmd: str):
    import subprocess
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTHONUNBUFFERED"] = "1"  # live logs on Modal (else stdout buffers ~6h)
    subprocess.run("cd /root/repo && git pull", shell=True, check=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/root/repo/experiments/ttt_trl")
    hf_cache.commit()
    out_vol.commit()

@app.function(image=image, secrets=[modal.Secret.from_name("zai")])
def verify_zai():
    """Cheap CPU-only check that the zai secret + key actually work before a GPU run."""
    import json
    import os
    import urllib.request

    key = os.environ.get("ZAI_API_KEY", "")
    print(f"[verify] ZAI_API_KEY present={bool(key)} len={len(key)}")
    body = json.dumps({
        "model": "glm-4.5-flash",
        "messages": [{"role": "user", "content": "reply with the single word OK"}],
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.z.ai/api/paas/v4/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "ttt-sdpo-judge/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print("[verify] ZAI OK ->", r.read().decode("utf-8")[:300])
    except Exception as e:  # noqa: BLE001
        print("[verify] ZAI FAILED ->", repr(e))


@app.local_entrypoint()
def main(cmd: str):
    call = run.spawn(cmd)
    print(f"Spawned! call id = {call.object_id}")
    print("Safe to close the laptop now — the function runs server-side on Modal.")