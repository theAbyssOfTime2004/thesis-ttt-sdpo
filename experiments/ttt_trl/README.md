# TTT-TRL Week 1 (Stage 1 Sanity)

This folder contains the minimum setup to verify TRL SDPO primitives on local GTX 1660 Ti 6GB.

## Goal (Week 1 / Stage 1)

Run 1+ SDPO training steps with `Qwen/Qwen2.5-0.5B-Instruct` without crash, and observe finite loss.

## Files

- `requirements.txt`: exact dependency pins for this stage.
- `env_check.py`: import/version/CUDA/VRAM smoke check.
- `00_trl_example_dummy.py`: dummy-data SDPOTrainer sanity run (no LCBv6, no LoRA yet).

## Setup (WSL + conda)

```bash
conda create -n ttt_trl python=3.11 -y
conda activate ttt_trl
```

Before install, verify GPU is visible inside WSL:

```bash
nvidia-smi
```

Install dependencies:

```bash
pip install -r experiments/ttt_trl/requirements.txt
```

Run environment check:

```bash
python experiments/ttt_trl/env_check.py
```

Expected output includes:

- `torch.cuda.is_available() == True`
- `trl.__version__ == '1.4.0'`
- `VRAM free >= 5.5 GB: True`

## Dummy SDPO run

```bash
python experiments/ttt_trl/00_trl_example_dummy.py
```

Expected:

- starts `SDPOTrainer`
- trains for 2 steps
- finite loss in logs (no NaN/inf)
- saves checkpoint under `outputs/00_dummy/final`

## Troubleshooting

- GTX 1660 Ti does not support bf16: keep `fp16=True`, `bf16=False`.
- If OOM occurs:
  - reduce `num_generations` from `4` to `2`
  - reduce max completion length to `64` (script includes adaptive fallback)
- TRL experimental API can change:
  - script introspects `SDPOConfig` signature and only passes supported args
  - if needed, print signatures with `inspect.signature(SDPOConfig)` and `inspect.signature(SDPOTrainer)`
