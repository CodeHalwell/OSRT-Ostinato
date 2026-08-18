# midtrain3 on Colab H100 — via the `colab` CLI (headless / agent-driven)

Colab gives the **same H100** as Modal, so `MidtrainExtend3Config` runs as-is.
The `colab` CLI (`uv tool install google-colab-cli`) drives it from the terminal.
Cross-session persistence is via a **private HF repo** (`HallD/osrt-v6-ckpt`):
`lightning_midtrain3.py --hf-repo` pulls the latest ckpt on start and pushes each
new one as it saves — necessary because the Colab VM disk is ephemeral and there's
a 24h/session cap (`scripts/hf_ckpt_sync.py`).

## One-time
```bash
# 1. auth the CLI (BROWSER — user step; ADC is the agent-friendly strategy)
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab whoami          # verify email + the 4 scopes

# 2. seed the HF ckpt repo with the base (from the Mac, ~4.9GB, once)
HF_TOKEN=… huggingface-cli upload HallD/osrt-v6-ckpt \
  checkpoints/v5/osrt_v5_midtrain2_step_1750.pt osrt_v5_midtrain2_step_1750.pt \
  --repo-type=model --private
```

## Per session (repeat until 550 units run out, ~2 sessions of ~24h)
```bash
colab new -s mt3 --gpu H100
colab status -s mt3                                   # confirm H100

# set up the VM (clone repo, deps, secrets) — piped shell via console
colab console -s mt3 <<'EOF'
cd /content && rm -rf osrt && git clone https://github.com/CodeHalwell/OSRT-605M-A269M.git osrt
cd osrt && uv pip install --system -q transformers==5.3.0 datasets==4.6.1 \
  tokenizers==0.22.2 safetensors==0.7.0 wandb==0.25.1 lion-pytorch==0.2.4 huggingface_hub
export HF_TOKEN=… WANDB_API_KEY=…
EOF

# SANITY GATE (30 steps) — run once, first session, before the burst
colab exec -s mt3 <<'EOF'
import os; os.chdir('/content/osrt'); os.environ['PYTHONPATH']='src'
os.environ['HF_TOKEN']='…'; os.environ['WANDB_API_KEY']='…'
os.system('python scripts/lightning_midtrain3.py --sanity --ckpt-dir /content/ckpt --hf-repo HallD/osrt-v6-ckpt')
EOF

# FULL BURST — background it on the VM so the local CLI can detach; poll via colab
colab console -s mt3 <<'EOF'
cd /content/osrt && export PYTHONPATH=src HF_TOKEN=… WANDB_API_KEY=…
nohup python scripts/lightning_midtrain3.py --ckpt-dir /content/ckpt \
  --hf-repo HallD/osrt-v6-ckpt --ckpt-interval 500 > /content/mt3.log 2>&1 &
EOF

# monitor (from local, anytime)
colab console -s mt3 <<'EOF'
tail -n 30 /content/mt3.log
EOF
# ...and the osrt-v6-midtrain3 W&B run for the ppl trend.

colab stop -s mt3          # ALWAYS stop when the session ends (idle VMs burn units)
```

On the **next** session: `colab new` again, re-run the VM setup, then just the
FULL BURST block — `--hf-repo` pulls the latest `midtrain3_step_*` from HF, the
resume-scan continues the same 12,600-step cosine. Keep going across sessions /
monthly Modal drip until ~1× Chinchilla, then re-run SFT v2 (EOS-fixed) on the
stronger base.

> Replace `…` with real `HF_TOKEN` / `WANDB_API_KEY`. Don't commit them.
