# Running the real test on Qwen-2.5-0.5B (RTX 3060, 6 GB, Windows)

Goal: test, at real-model scale, (1) whether Adam's momentum leak increases forgetting vs SGD and
(2) whether group-lasso slot selection beats KL / TF-IDF. Scaled down from the paper so it fits about
2–4 hours of GPU time (my estimate; not measured on your machine).

## 1. One-time setup (PowerShell, in your home folder)
```powershell
cd $HOME
git clone https://github.com/prakharg55/SMF-ICML-FG smf
cd smf
git apply $HOME\gemma-lasso-router\continual_learning\smf_glasso_sgd.patch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line should print `True NVIDIA GeForce RTX 3060 ...`.

## 2. Retrofit memory layers (once, est. 30–90 min)
```powershell
$env:RUN = "$HOME\smf_runs"
python -m smf_rebuild.train_retrofit `
  --output-dir "$env:RUN\retrofit" --memory-mode additive --memory-scale-init 0.01 `
  --layers 6,12,18 --mem-n-keys 128 --mem-heads 4 --mem-knn 16 --mem-k-dim 256 `
  --dataset-preset oasst1 --sample-size 10000 --eval-sample-size 500 --max-length 512 `
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 --num-train-epochs 1 `
  --dtype bf16 --dataloader-num-workers 0 --num-proc 1 --report-to ""
```
If you get an out-of-memory error, add `--max-length 256`.

## 3. Sparse finetuning runs (est. 10–25 min each)
The shared settings are stored in `$common` below. Run one line at a time, from the same PowerShell window.
```powershell
$common = @("--init-checkpoint", "$env:RUN\retrofit", "--dataset-preset", "medmcqa",
  "--sample-size", "5000", "--eval-sample-size", "300", "--max-length", "256",
  "--background-preset", "oasst1", "--background-sample-size", "2000", "--background-max-batches", "500",
  "--trainable-scope", "values", "--per-device-train-batch-size", "4", "--gradient-accumulation-steps", "4",
  "--num-train-epochs", "1", "--dtype", "bf16", "--dataloader-num-workers", "0", "--num-proc", "1", "--report-to", "")

# A. Their default: KL + AdamW (has the momentum leak)
python -m smf_rebuild.train_sparse @common --output-dir "$env:RUN\kl_adam" --sparse-scoring kl --sparse-top-t 512 --optim adamw --learning-rate 5e-4
# B. KL + SGD (no leak)
python -m smf_rebuild.train_sparse @common --output-dir "$env:RUN\kl_sgd" --sparse-scoring kl --sparse-top-t 512 --optim sgd --learning-rate 2.0
# C. TF-IDF + SGD
python -m smf_rebuild.train_sparse @common --output-dir "$env:RUN\tfidf_sgd" --sparse-scoring tfidf --sparse-top-t 512 --optim sgd --learning-rate 2.0
# D. Gary's group lasso + SGD
python -m smf_rebuild.train_sparse @common --output-dir "$env:RUN\glasso_sgd" --sparse-scoring glasso --sparse-lam 0.5 --optim sgd --learning-rate 2.0
```
The SGD learning rate of 2.0 follows Lin et al.'s range (2–5). If medical accuracy doesn't move at all,
retry B–D with `--learning-rate 5.0`.

## 4. Evaluate each (est. 5–15 min each)
```powershell
foreach ($r in "retrofit","kl_adam","kl_sgd","tfidf_sgd","glasso_sgd") {
  python -m smf_rebuild.eval_tasks --memory-checkpoint "$env:RUN\$r" `
    --tasks medmcqa,wikitext,triviaqa --limit 500 --output "$env:RUN\eval_$r.json"
}
```
Then send me the `eval_*.json` files (or point me at the folder) and I'll analyze them.

What to look for:
- MedMCQA up = learning
- WikiText perplexity up and TriviaQA down = forgetting
- `retrofit` is the before-picture
