# Training Pipeline Improvement Report

**Audit date:** 2026-07-23  
**Scope:** dataset preparation, QLoRA training, checkpoint evaluation, the pinned WSL environment, and reproducibility.  
**Audit mode:** read-only. No pipeline or dataset files were changed during this audit.

## Executive summary

The individual components are promising, but the end-to-end pipeline is **not release-ready or reproducible yet**. The WSL/GPU environment, tokenizer format, and current data-builder scripts work in isolation. The workflow can nevertheless train stale data, accept an incomplete dataset without failing, omit validation entirely, and fail during evaluation with the currently installed Transformers API.

The highest-priority work is to make the dataset a verified, versioned input; make quality-control failures blocking; wire validation into training; and fix and lock the inference dependency contract.

## What was verified

| Area | Result | Evidence |
|---|---|---|
| WSL hardware stack | Pass | Ubuntu 24.04, RTX 2060 12 GB, Torch 2.6.0+cu126, Triton 3.2.0, bitsandbytes 0.49.2; `training/sanity_check.py` completed the NF4 fp16 forward-pass check. |
| Static syntax | Pass | Python sources compiled successfully and `bash -n training/setup_wsl.sh` passed. |
| Existing final dataset | Structurally valid | 1,670 train and 89 validation rows; valid two-message schema; no exact pair, prompt, or completion overlap between splits. |
| Length and template compatibility | Pass for the existing split | Qwen's template contains the response-mask delimiters used by training. Existing examples are 49–289 chat tokens, all below the 1,024-token limit. |
| Dataset scripts in isolation | Operational, but not gated | An offline temporary-copy run of the current validator and builder produced 1,979 accepted / 21 rejected records and an 1,880 / 99 train/validation split. |
| Full training and generation | Not run | A real training run would create expensive model artifacts. Static and live API checks exposed blockers that should be fixed first. |

The 80-probe set has no exact prompt or ID overlap with the source training prompts. That is a good baseline, but it should become an automated check.

## Current flow and weak points

```text
manifests + raw generations
        |
        v
validate.py  -- currently reports failures but exits successfully
        |
        v
accepted.jsonl -> dedup.py -> build_train_jsonl.py
        |                              |
        |                              v
        |                       final/train.jsonl + final/val.jsonl
        |                                      |
        +--------------------------------------v
                                   train.py
                                      |
                                      v
                              checkpoints -> eval/generate.py
```

The current `train.py` consumes pre-built `dataset/final/*.jsonl` files directly. It does not prove that they were generated from the current raw data, that they met quotas, or that the configured validation split is used.

## Priority 0 — fix before a production-quality run

### 1. Reconcile and version the training dataset

**Finding**

The checked-in/generated state is internally stale:

- The manifests contain 2,000 unique IDs.
- The raw directory contains 2,241 rows: 2,000 IDs plus 241 `regen_*` replacements.
- The current validator accepts 1,979 rows after applying those replacements; 21 remain rejected.
- The existing `dataset/generated/accepted.jsonl` and `dataset/final/*.jsonl` contain only 1,759 rows. They predate the regeneration wave, omitting 220 now-accepted replacements.
- `dataset/final/` and the `regen_*` raw files are currently untracked. A clean checkout cannot run `train.py` as configured until it reconstructs or fetches the final files.

**Impact**

A training run can silently use an older dataset than the one reviewers believe they approved. The data cannot be reproduced from a clean clone without undocumented manual work.

**Required improvement**

1. Decide the canonical snapshot: regenerate the final files from the current raw data only after the QC policy below is satisfied.
2. Make the source-of-truth policy explicit. Either version the final JSONL plus its provenance in Git, or version raw/manifests and provide a deterministic, documented build step that produces final JSONL.
3. Store a dataset manifest with at least: accepted count, per-archetype counts, source-file hashes, final-file hashes, tokenizer/model revision, seed, and generation timestamp.
4. Have training verify the manifest and final-file hashes before starting, or make the build step part of the training entry point.

**Acceptance criteria**

- A clean checkout can build or fetch the exact final dataset without hand edits.
- `accepted.jsonl`, `train.jsonl`, and `val.jsonl` agree on a documented count and hash.
- Each accepted record has a traceable manifest/raw source ID.

### 2. Turn dataset QC into a real gate and resolve the quota contract

**Finding**

[`dataset/scripts/validate.py`](dataset/scripts/validate.py) writes its outputs at lines 133–140, then only prints missing, rejection, and per-archetype counts at lines 142–151. It exits with status zero even when records are rejected or quotas are missed.

The stated quota contract also disagrees with the manifests:

| Archetype | Config target | Manifest count | Current validator result |
|---|---:|---:|---:|
| `competent_then_pivot` | 400 | 410 | 404 accepted |
| `small_talk` | 60 | 50 | 50 accepted |

The 10-record rebalance may be intentional, but it is not represented consistently in configuration, manifests, or validation behavior.

**Impact**

The word “gate” is misleading: automation can train a reduced or skewed data distribution after a successful command exit. This makes model behavior and paper statistics difficult to trust.

**Required improvement**

1. Choose one policy for rejected examples: regenerate until every intended manifest ID has an accepted replacement, or explicitly permit an approved shortfall per archetype.
2. Encode the final target counts in one source of truth. If rebalance is valid, update the target plan; if it is not, fail manifest generation.
3. Make `validate.py` return non-zero when any required ID is missing, an unapproved shortfall exists, an unknown ID appears, or a configured quality threshold is breached.
4. Make `build_train_jsonl.py` refuse to build from a failed/unsigned QC result.
5. Emit a machine-readable summary (`qc_summary.json`) rather than relying solely on console output.

**Acceptance criteria**

- Intentionally malformed, missing, or quota-short fixture data causes a non-zero exit.
- A valid data release produces a signed summary with zero unapproved failures.
- The final split exactly reflects the approved per-archetype plan.

### 3. Use the validation split during training

**Finding**

[`training/train.py`](training/train.py) loads `train` and `val` at lines 83–86, but only maps and supplies `train_ds` to `SFTTrainer` at line 118. Its `SFTConfig` does not set an evaluation strategy, whose current default is `"no"`.

**Impact**

There is no validation loss, no overfitting signal, and no training-time proof that the validation data was even readable. This also conflicts with the paper's `{{VAL_LOSS}}` placeholder.

**Required improvement**

1. Apply the same chat-template transform to the validation dataset.
2. Pass it as `eval_dataset` to `SFTTrainer`.
3. Configure an explicit evaluation cadence (`steps` or `epoch`) and log validation loss alongside training loss.
4. Keep behavioral probe evaluation and human review as the final checkpoint-selection method; validation loss should be a health signal, not the sole quality objective.
5. Document the checkpoint-selection rule, including the role of pivot rate, engagement, leakage, diversity, and human review.

**Acceptance criteria**

- A smoke run logs at least one validation-loss measurement.
- The trainer rejects a missing or malformed validation file before training progresses.
- A run report records the chosen checkpoint and the selection rationale.

### 4. Fix the active inference API incompatibility

**Finding**

The currently installed environment uses `transformers==5.5.0`. Its Qwen tokenizer returns a `BatchEncoding` by default for this call:

```python
tokenizer.apply_chat_template(..., return_tensors="pt")
```

The exact live probe showed that this object has no `.shape`. The following scripts subsequently access `ids.shape`, so they will raise `AttributeError` under the current dependency set:

- [`training/train.py`](training/train.py) smoke generation at lines 131–136
- [`eval/generate.py`](eval/generate.py) at lines 56–72
- [`training/chat.py`](training/chat.py) at lines 55–66

**Impact**

Training can complete its optimization loop but smoke output, checkpoint evaluation, and interactive chat are not reliable. That removes the mechanism needed to select and inspect checkpoints.

**Required improvement**

Use one explicit, tested representation everywhere:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
).to("cuda")
```

Alternatively, retain the `BatchEncoding` and use `inputs["input_ids"]` consistently for both `model.generate` and output slicing. Apply the choice to smoke generation, evaluation, and chat together.

**Acceptance criteria**

- A tokenizer API test asserts the expected input type and output slicing behavior.
- `train.py --smoke`, `eval/generate.py --base`, and `chat.py` each reach a decoded response without an API/type error.

## Priority 1 — make results reproducible and operable

### 5. Lock the complete dependency and model contract

**Finding**

[`training/setup_wsl.sh`](training/setup_wsl.sh) pins Torch and Triton but installs Unsloth, TRL, Transformers, datasets, and bitsandbytes transitively without exact version constraints. The live environment resolved to:

```text
unsloth        2026.7.4
unsloth-zoo    2026.7.4
transformers   5.5.0
trl            0.24.0
datasets       4.3.0
bitsandbytes   0.49.2
```

`pip check` currently fails because `unsloth-zoo` declares a dependency on `torchao`, while the setup script deliberately removes it. The current API regression demonstrates why partial pinning is insufficient. The Qwen model and tokenizer also use an unpinned Hub revision.

**Required improvement**

1. Create a committed lock/constraints file containing every package version validated on the RTX 2060.
2. Resolve the `torchao` conflict rather than accepting a failed `pip check`: select a compatible Unsloth release, document a supported optional-dependency exemption, or use an installation combination whose metadata is consistent.
3. Pin the model and tokenizer to an immutable Hub revision in configuration.
4. Make `pip check`, package-version capture, and a tokenizer API test part of setup validation.
5. Save `pip freeze`, GPU/driver details, config, Git commit, and dataset manifest with every run.

**Acceptance criteria**

- A fresh WSL virtual environment installs from the lock file with `pip check` passing.
- The complete smoke/evaluation flow passes on the locked versions.
- Repeating setup after upstream releases resolves to the same package and model revisions.

### 6. Remove machine-specific path assumptions

**Finding**

The repository path is hard-coded in [`training/config.yaml`](training/config.yaml), [`training/setup_wsl.sh`](training/setup_wsl.sh), [`eval/generate.py`](eval/generate.py), and [`training/sync_outputs.sh`](training/sync_outputs.sh). The current paths assume one Windows username, one checkout location, and one WSL mount.

**Required improvement**

1. Resolve repository-relative dataset paths from the script location.
2. Keep only intentionally external locations configurable, such as the ext4 run directory and Hugging Face cache.
3. Give each run a unique ID and persist its configuration rather than treating `run01` as a reusable mutable directory.
4. Fail early with a clear message when a resolved path does not exist.

**Acceptance criteria**

- The project works from a different checkout path without edits.
- A second run cannot accidentally overwrite or mix artifacts from the first.

### 7. Add layered tests and CI checks

**Required improvement**

Add inexpensive CPU checks plus a separately tagged WSL/GPU smoke check.

| Layer | Minimum coverage |
|---|---|
| Unit | ID replacement semantics, reject policy, quota policy, split disjointness, template markers, and inference-input type handling. |
| Data integration | Validate → deduplicate → build final split from a small fixture; assert counts, hashes, and non-zero failure cases. |
| Training smoke | Load the pinned tokenizer/model, run a few steps, check response-only masking, log validation loss, and save a usable checkpoint. |
| Evaluation smoke | Generate one base and one adapter response, produce metrics, and render a report. |
| CI | Run static checks and CPU fixture tests on every change; run the GPU smoke test before a data/model release. |

## Recommended implementation order

1. **Make a policy decision:** approve revised quotas and 21 remaining rejects, or regenerate until the intended dataset is complete.
2. **Fix data release mechanics:** make validation blocking, rebuild the final split, record hashes/provenance, and version or deterministically rebuild those artifacts.
3. **Fix runtime correctness:** wire validation into `train.py` and correct the `BatchEncoding` handling in all generation paths.
4. **Freeze the environment:** add the dependency lock, model revision, successful `pip check`, and portable paths.
5. **Add tests:** start with data-QC fixtures and tokenizer/inference API coverage, then add the GPU smoke path.
6. **Run a release candidate:** run the complete chain from a fresh environment; retain the generated run manifest, checkpoints, metrics, and human-review report.

## Definition of done

Do not call the pipeline robust until all of the following are true:

- [ ] A clean checkout and documented setup reproduce the exact dataset and environment.
- [ ] QC fails closed on unapproved data defects and quota drift.
- [ ] The final train/validation split has a signed, traceable provenance manifest.
- [ ] Training evaluates the validation split and records its metrics.
- [ ] Smoke, base-model evaluation, adapter evaluation, and chat all execute under the locked environment.
- [ ] Checkpoint selection follows a documented behavioral and human-review rule.
- [ ] A fresh setup passes `pip check` and the test suite.

## Decisions needed from the project owner

1. Should the release target be exactly 2,000 accepted examples, or is a documented 1,979-example dataset acceptable?
2. Is the intended quota 400/60 for `competent_then_pivot`/`small_talk`, or is the 410/50 rebalance the new target?
3. Should generated final datasets live in Git, in a versioned artifact store, or be rebuilt deterministically from committed raw data?
4. Which behavioral metrics are hard checkpoint-selection gates, and which are advisory for human review?

Once those decisions are made, the required code changes are localized; they do not require an architectural rewrite.
