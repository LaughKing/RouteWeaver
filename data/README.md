# Datasets

The repository ships code only. Everything the training and evaluation scripts
read lives under `$ROUTEWEAVER_DATASETS` (default `<repo>/datasets`, which is
gitignored).

```bash
python data/hf_download.py
```

```
datasets/
  parquet/     routeweaver_train.parquet       the 3,200-question training set
               routeweaver_cold_pool.parquet   the cold start's own pool
               val_stub.parquet                the 32-row stub verl requires
  eval/        eval_600_free, eval_700_free, eval_triviaqa120_free
               ood_{aime149,gpqa_diamond198,bbh1080}_free
  cost_calib/  cost_calib_128.parquet          the C_ref calibration set
```

Every question is stored as a verl prompt row: the mode menu and the six
anonymous worker ids are **baked in**, so a prompt cannot drift with the code,
and `extra_info.batch_index` is the training step the row belongs to — the
loader reads the file with `shuffle=False`, so row order IS the schedule. The
dataset card documents the row format, the per-file composition and the
sources.

Point the scripts at your own files by setting `ROUTEWEAVER_DATASETS`, or
override `TRAIN_FILES` / `VAL_FILES` per run.
