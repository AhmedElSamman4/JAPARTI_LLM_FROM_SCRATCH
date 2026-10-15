<div align="center">

# 🌙 Jabarti LLM

**A bilingual Arabic + English GPT, built from scratch.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Package manager](https://img.shields.io/badge/uv-managed-purple)
![Platform](https://img.shields.io/badge/Ubuntu-WSL-orange)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

</div>

---

## ✨ What is this?

Jabarti LLM is a learning project where I build a GPT-style language model step by step, with no shortcuts:

**Clean data → Train tokenizer → Pretrain → Chat fine-tune (LoRA) → Evaluate**

Everything is written to be read, run, and understood.

---

## 🗺️ Roadmap

| # | Stage | Status |
|---|-------|--------|
| 1 | PyTorch intro | ✅ Done |
| 2 | Explore the dataset | ✅ Done |
| 3 | Clean data + build the BPE tokenizer | ✅ Done |
| 4 | Pretrain the GPT model | 🚧 Planned |
| 5 | Chat fine-tuning with LoRA | 🚧 Planned |
| 6 | Evaluation and monitoring | 🚧 Planned |

---

## 📁 Project structure

```text
JAPARTI_LLM_FROM_SCRATCH/
├── scripts/
│   ├── CH01_pytorch_intro/
│   ├── CH02_explore_dataset/
│   └── CH03_build_tokenizer/
│       ├── clean_corpus.py        # Step 1: raw data → cleaned data
│       ├── cleaning.py            # Cleaning / normalization rules
│       ├── build_tokenizer.py     # Step 2: train the BPE tokenizer
│       ├── tokenizer.json         # Result: the trained tokenizer
│       └── output/                # Parquet datasets (raw, cleaned, fine-tuning)
├── pyproject.toml                 # Dependencies
├── uv.lock                        # Locked versions
├── .python-version                # Python 3.11
└── README.md
```

<details>
<summary><b>What's inside <code>output/</code>?</b></summary>

| File | Meaning |
|------|---------|
| `phase1_train.parquet`, `phase1_valid.parquet` | Raw phase 1 data |
| `phase1_train_cleaned.parquet`, `phase1_valid_cleaned.parquet` | Cleaned phase 1 data |
| `phase2_train.parquet`, `phase2_valid.parquet` | Raw phase 2 data |
| `phase2_train_cleaned.parquet`, `phase2_valid_cleaned.parquet` | Cleaned phase 2 data |
| `ft_train.parquet`, `ft_valid.parquet` | Chat fine-tuning data |

The tokenizer reads **only the `_cleaned` files**.

</details>

---

## 🚀 Quick start

### 1. Set up Ubuntu on Windows (skip if you already have it)

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart if asked, then open **Ubuntu** from the Start menu and create your username and password.

### 2. Open the project in Ubuntu

```bash
cd /mnt/c/My_github/JAPARTI_LLM_FROM_SCRATCH
```

> Change the path to where your repository actually is.

> ⚠️ Run all project commands from the **Ubuntu terminal**, not PowerShell.

### 3. Install the tools

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl
```

Install [`uv`](https://docs.astral.sh/uv/) (follow the official instructions), then check:

```bash
uv --version
```

### 4. Create the environment and install the project

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

You are ready. 🎉

---

## 🔤 Build the tokenizer

The tokenizer is trained in two steps. Go to the chapter folder first:

```bash
cd scripts/CH03_build_tokenizer
```

**Step 1: Clean the raw data**

```bash
python clean_corpus.py
```

This creates the `*_cleaned.parquet` files in `output/`.

**Step 2: Train the tokenizer**

```bash
python build_tokenizer.py
```

This reads the cleaned files and saves `tokenizer.json`.

### Tokenizer settings

| Setting | Value |
|---------|-------|
| Algorithm | BPE (Byte Pair Encoding) |
| Vocabulary size | 32,000 |
| Minimum token frequency | 2 |
| Pre-tokenization | Metaspace |
| Normalization | Rules from `cleaning.py` |

### Special tokens

```text
[PAD] [UNK] [BOS] [EOS] [SEP]
[AR]  [EN]  [SYS] [USER] [ASST]
```

### Sanity check

After training, the script is used to check Arabic and English examples: input text, tokens, token IDs, and decoded output. A good tokenizer should decode back to the original text.

> A sanity check is not a full quality test. The real test is how the model behaves after training.

---

## 🧠 Next stages (planned)

> These scripts are not in the repository yet. The commands below are the **plan**, so check them against the real scripts when they are added.

<details>
<summary><b>Pretraining</b></summary>

Teach the model to predict the next token.

Start with a small test run to check the pipeline:

```bash
python run_pretrain.py --steps 60 --warmup 10 --limit 20000 --eval-limit 2000 \
    --batch-size 32 --accumulation-steps 4 --lr 6e-4 --shuffle-seed 42
```

Then run the full training:

```bash
python run_pretrain.py --epochs 15 --batch-size 56 --accumulation-steps 4 \
    --lr 6e-4 --warmup 500 --weight-decay 0.1 \
    --checkpoint-dir ../checkpoints/jabarti-512x8-ep15 --save-every 100
```

Expected output: `pretrain_final.pt`

> Lower `--batch-size` if you run out of GPU or system memory.

</details>

<details>
<summary><b>LoRA chat fine-tuning</b></summary>

Adapt the pretrained model to follow chat conversations. LoRA trains small adapter matrices instead of every model weight.

```bash
python run_finetune_lora.py \
    --resume ../checkpoints/jabarti-512x8-ep15/pretrain_final.pt \
    --epochs 100 --batch-size 56 --accumulation-steps 4 \
    --lr 2e-5 --warmup 100 --lora-r 256 --lora-alpha 512 \
    --checkpoint-dir ../checkpoints/jabarti-512x8-lora
```

Expected output: `finetune_chat_lora_final.pt` (adapters merged into the model)

| Option | Meaning |
|--------|---------|
| `--lora-r` | Rank of the adapter. Higher means more capacity. |
| `--lora-alpha` | Scaling of the update (`alpha / r`). |

</details>

<details>
<summary><b>Monitor training with Trackio</b></summary>

```bash
trackio show --project jabarti-llm --host 0.0.0.0
```

Open `http://localhost:7860` in your Windows browser.

To view it from another device, you can use a temporary Cloudflare tunnel:

```bash
cloudflared tunnel --protocol http2 --url http://localhost:7860
```

> 🔒 The tunnel URL is public. Use it only when needed and stop it when you are done.

</details>

---

## 🧰 Troubleshooting

| Problem | Fix |
|---------|-----|
| `python: command not found` | Activate the environment: `source .venv/bin/activate` |
| `uv: command not found` | Reinstall `uv` and reopen the terminal |
| File or path not found | Check your folder with `pwd` and `ls -la` |
| Out of memory | Reduce the batch size |

---

## 📌 Notes

- Large files (`.pt` checkpoints, generated datasets) should stay out of git.
- Results depend on your data, tokenizer, model size, and hardware.

---

<div align="center">

Built as a learning journey into how language models really work. 🌙

</div>