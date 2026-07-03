# Modal runner for TTT-SDPO

Minimal wrapper (`modal_run.py`) that runs existing scripts on a Modal **A100-80GB** GPU. No training logic lives here — it only builds the image, mounts volumes/secrets, and executes a subprocess in `experiments/ttt_trl/`.

## Secrets

Create three Modal secrets (Dashboard → Secrets, or `modal secret create`):

| Secret name   | Required env var   | Purpose |
|---------------|--------------------|---------|
| `wandb`       | `WANDB_API_KEY`    | Weights & Biases logging |
| `zai`         | `ZAI_API_KEY`      | LLM judge (z.ai / GLM) |
| `huggingface` | `HF_TOKEN`         | HF model/dataset access (must accept gated `google/gemma-4-E4B-it`) |

Example:

```bash
modal secret create wandb WANDB_API_KEY=...
modal secret create zai ZAI_API_KEY=...
modal secret create huggingface HF_TOKEN=...