# Wav2Vec2 Fine-Tuning Method

This project fine-tunes `facebook/wav2vec2-base` with CTC using LibriSpeech-based WebDataset shards, then checks inference outputs and WER on `test-clean` and `test-other`.

## Overall Flow

1. `run/sample_util.py` reads WebDataset shards and preprocesses audio/text samples.
2. `run/wav2vec_finetuning.py` attaches a CTC head to the Wav2Vec2 base checkpoint and fine-tunes it.
3. The fine-tuned model and processor are saved under `finetuning_output/`.
4. `run/wav2vec_inference.py` transcribes `test-clean` and `test-other` with the saved model.
5. `run/evaluate_wer.py` computes WER from REF/HYP result files.

## Data Layout

Place the data under `data/` at the project root.

```text
data/
  1h/
    shard-000000.tar
    ...
    shard-000004.tar
  test-clean/
    shard-000000.tar
    ...
    shard-000004.tar
  test-other/
    shard-000000.tar
    ...
    shard-000005.tar
```

Each sample inside a tar shard has the following extensions.

```text
<sample-id>.audio
<sample-id>.text
<sample-id>.meta
```

The code sorts and reads all `*.tar` files, and only uses `audio` and `text` for training and evaluation. WebDataset is configured with `shardshuffle=False` to keep shard order fixed.

During fine-tuning, `test-clean` and `test-other` are not used as validation sets. They are only used for final inference and WER computation. The validation set for model selection is split from the shards inside `data/1h`.

With the default setting (`--val-ratio 0.2`), the last 1 shard out of the 5 shards in `data/1h` is used as validation.

```text
data/1h/
  shard-000000.tar  -> train
  shard-000001.tar  -> train
  shard-000002.tar  -> train
  shard-000003.tar  -> train
  shard-000004.tar  -> validation
```

The validation split is only used to compute WER every `eval_steps` during training and to select the best checkpoint. Even when training augmentation is enabled, augmentation is not applied to validation.

## Preprocessing

`run/sample_util.py` converts each sample as follows.

- Decode audio bytes with `soundfile` first, and fall back to `torchaudio.load()` if that fails.
- Convert the waveform to a CPU float tensor.
- Convert stereo or multi-channel audio to mono.
- Resample audio to 16 kHz with `torchaudio.functional.resample()` when needed.
- Read transcripts as UTF-8 text, normalize whitespace, and convert to uppercase.
- Convert audio to `input_values` with the processor feature extractor.
- Convert transcripts to CTC label IDs, `labels`, with the processor tokenizer.
- Also return `speech`, the raw 16 kHz NumPy array, and `sampling_rate` for inference.

The current checkpoint settings are:

```text
MODEL_NAME = facebook/wav2vec2-base
PROCESSOR_NAME = facebook/wav2vec2-base-960h
```

`facebook/wav2vec2-base` is a pretraining checkpoint and may not provide an ASR tokenizer directly. Therefore, the acoustic model uses the base checkpoint, while CTC label creation and decoding use the `facebook/wav2vec2-base-960h` processor, which includes an ASR tokenizer.

## Fine-Tuning

`run/wav2vec_finetuning.py` loads `facebook/wav2vec2-base` as a CTC model through `AutoModelForCTC.from_pretrained()`. Since the base checkpoint is a pretraining model without an ASR CTC head, `lm_head` is newly initialized. By contrast, quantizer/projection weights used for pretraining are not used for CTC training.

The key model loading settings are:

```text
apply_spec_augment = False
ctc_zero_infinity = True
ignore_mismatched_sizes = True
use_safetensors = True
vocab_size = len(processor.tokenizer)
```

`apply_spec_augment=False` is a stability setting to avoid NaN loss/gradients in train mode. `ctc_zero_infinity=True` prevents infinite CTC losses from breaking gradients. `use_safetensors=True` is used to load `facebook/wav2vec2-base` safely in the current torch environment.

## Batch Construction and CTC Label Padding

Wav2Vec2 CTC training uses dynamic padding because audio length and transcript length differ across samples.

`DataCollatorCTCWithPadding` does the following:

- Pads batch audio `input_values` to the longest audio length in the batch with `processor.pad()`.
- Pads label token IDs to the longest label length in the batch with `processor.tokenizer.pad()`.
- Replaces label padding positions with `-100`.

Label padding is replaced with `-100` so that PyTorch CTC loss ignores those positions. During evaluation, `-100` is converted back to the tokenizer pad token ID before decoding text.

## Training Settings

The default training settings for an A5000 24GB GPU are:

```text
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
effective_batch_size = 32
per_device_eval_batch_size = 8
learning_rate = 1e-4
warmup_steps = 100
max_steps = 2000
gradient_checkpointing = True
fp16 = False by default
tf32 = True on CUDA
```

The feature encoder is frozen with `model.freeze_feature_encoder()`. For small-data fine-tuning such as the 1h setup, freezing the low-level acoustic feature extractor improves training stability.

fp16 is disabled by default, so training uses fp32 + TF32. Use the `--fp16` argument to experiment with fp16.

## Evaluation

During fine-tuning, the validation shard split from `data/1h` is used as the eval dataset. With the default setting, `shard-000004.tar` is the validation set. Token IDs are selected from model output logits with `argmax`, then decoded with the processor. Reference labels are decoded after restoring `-100` to the pad token ID. WER is then computed with `evaluate.load("wer")`.

The best checkpoint is selected based on the model with the lowest validation WER.

```text
metric_for_best_model = wer
greater_is_better = False
```

`test-clean` and `test-other` are not used as eval datasets or for checkpoint selection during fine-tuning. They are only used after training to compute final test WER.

## Inference

`run/wav2vec_inference.py` uses the fine-tuned model saved under `finetuning_output/` by default. If a trained model is not available yet, it falls back to the `PROCESSOR_NAME` checkpoint for baseline inference.

Default inference uses CTC greedy decoding without an external language model. If `--lm-model` is specified, inference runs neural LM shallow fusion decoding, which combines Hugging Face causal language model scores inside CTC prefix beam search.

Neural LM shallow fusion directly adds LM log-likelihood to candidate scores during beam search, affecting both prefix pruning and final selection. Calling OPT for every character prefix, as in the previous approach, would require LM forwards on the scale of `number of frames * beam width * token beam`, which is very slow. The current implementation improves speed by updating LM scores in batches only when word-boundary candidates appear, and scores each utterance's final candidates one more time before final selection.

When `--lm-model` is used, an internal CTC frame progress bar is shown under the dataset-level tqdm progress bar for each utterance. The internal progress bar displays the current beam size, the number of prefixes scored by the LM on the current frame, and the LM cache size. Use `--no-lm-progress` to disable the extra progress bars.

```text
score = ctc_score + lm_alpha * lm_score + word_bonus * word_count
```

When shallow fusion is used, prefix text has already been collapsed by CTC prefix beam search. Therefore, decoding uses `group_tokens=False` so repeated letters such as `LL` and `EE` are not collapsed again.

To improve speed, reduce `--beam-width` and `--token-beam`. Larger values increase the number of acoustic candidates and the number of LM candidates that must be batch-scored at word boundaries. `--lm-batch-size` controls the maximum batch size for neural LM scoring and defaults to `512`. If GPU memory is sufficient but utilization is low, increasing this value may help.

Results are saved at the project root.

```text
test_clean_result.txt
test_other_result.txt
```

Each file follows this format.

```text
REF: reference transcript
HYP: predicted transcript
```

## How to Run

Move to the project root and activate the conda environment.

```bash
cd ~/disk2/syju/code/SpeechRecognition_2026-1_FinalProject
conda activate syju_speech
```

Training:

```bash
python run/wav2vec_finetuning.py
```

Inference:

```bash
python run/wav2vec_inference.py
```

WER computation:
`run/evaluate_wer.py` reads REF/HYP files, creates reference and hypothesis lists, and computes WER with `jiwer.wer()`.

```bash
python run/evaluate_wer.py test_clean_result.txt
python run/evaluate_wer.py test_other_result.txt
```

## Notes

- When `facebook/wav2vec2-base` is loaded as a CTC model, `lm_head` appears as `MISSING`. This means the CTC output head is newly initialized and will be fine-tuned.
- Pretraining quantizer/projection weights appearing as `UNEXPECTED` is expected because those weights exist in the checkpoint but are not used for CTC training.
- If `loss=0` and `grad_norm=nan` appear together, the run is not training normally. Stop the run and check the default fp32 setting and `apply_spec_augment=False`.
- Major experiment options for the ablation study are controlled through command-line arguments.

## Current Code Change Summary

- `test-clean` and `test-other` are not used as eval datasets during fine-tuning.
- `data/1h` shards are split into train/validation, and the best checkpoint is selected by validation WER.
- With the default `--val-ratio 0.2`, `shard-000000.tar` through `shard-000003.tar` are used for training, and `shard-000004.tar` is used for validation.
- The previous N-best rescoring method has been replaced by neural LM shallow fusion.
- Shallow fusion is controlled by `--lm-model`, `--lm-alpha`, and `--word-bonus`.
- Neural LM scores are computed in batches at word-boundary candidates instead of every character prefix, improving inference speed.
- During LM decoding, utterance-level frame progress is displayed with tqdm, and `--lm-batch-size` controls the LM scoring batch size.
- Existing WER values need to be rerun because the split and decoding method have changed.

## Baseline Summary

The current baseline in `run/` starts from `facebook/wav2vec2-base` and performs CTC fine-tuning with the LibriSpeech 1h shards.

- Uses the `facebook/wav2vec2-base` checkpoint as the initial acoustic model.
- Uses a grapheme-based tokenizer/processor to convert transcripts into CTC labels.
- Attaches a CTC head through `AutoModelForCTC` and fine-tunes the model.
- Saves the fine-tuned model and processor under `finetuning_output/`.
- Runs inference on `test-clean` and `test-other` to create `test_clean_result.txt` and `test_other_result.txt`.
- Computes WER from REF/HYP result files with `run/evaluate_wer.py`.

## Baseline Results

```bash
python run/evaluate_wer.py test_clean_result.txt
# WER: TBD

python run/evaluate_wer.py test_other_result.txt
# WER: TBD
```

| Evaluation set | WER |
| --- | ---: |
| test-clean | TBD |
| test-other | TBD |

## Ablation Study Plan

The following experiments use the same train/validation split and compare `test-clean` and `test-other` WER after training. Running `python run/wav2vec_finetuning.py` without any ablation option uses the baseline setting. Keeping separate result file names and output directories makes experiment results easier to update.

| Experiment | Training option | Decoding option | test-clean WER | test-other WER | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Baseline | default CTC fine-tuning | greedy decoding | 0.2800 | 0.4035 | `wav2vec2 CTC greedy decoding` |
| Ablation 1 | `--augment` | greedy decoding | 0.2434 | 0.3417 | data augmentation |
| Ablation 2 | `--freeze-transformer-layers 6` | greedy decoding | 0.2439 | 0.3219 | frozen encoder layers |
| Ablation 3 | `--max-entropy-weight 0.01` | greedy decoding | 0.2553 | 0.3667 | maximum entropy regularization |
| Ablation 4 | default CTC fine-tuning | `--lm-model facebook/opt-125m` | 0.2720 | 0.3960 | neural LM shallow fusion |
| Ablation 1+2+3 | `--augment` + `--freeze-transformer-layers 6` + `--max-entropy-weight 0.01` | greedy decoding | 0.2316 | 0.3005 | combined training |
| Ablation 1+2+3+4 | `--augment` + `--freeze-transformer-layers 6` + `--max-entropy-weight 0.01` | `--lm-model facebook/opt-125m` | 0.2239 | 0.2930 | combined training + decoding setting |

### Ablation Training Commands

The baseline is trained without additional options. Each ablation uses a different output directory so checkpoints and results remain separated.

```bash
# Baseline: wav2vec2 CTC greedy decoding
python run/wav2vec_finetuning.py \
  --output-dir finetuning_output_baseline

# Ablation 1: + data augmentation
python run/wav2vec_finetuning.py \
  --augment \
  --output-dir finetuning_output_aug

# Ablation 2: + frozen encoder layers
python run/wav2vec_finetuning.py \
  --freeze-transformer-layers 6 \
  --output-dir finetuning_output_freeze6

# Ablation 3: + maximum entropy regularization
python run/wav2vec_finetuning.py \
  --max-entropy-weight 0.01 \
  --output-dir finetuning_output_maxent001
```

Neural LM shallow fusion is a decoding ablation, not a training ablation, so it is applied during inference with the baseline checkpoint.

```bash
# Ablation 4: + neural LM shallow fusion
python run/wav2vec_inference.py \
  --model-dir finetuning_output_baseline \
  --lm-model facebook/opt-125m \
  --beam-width 25 \
  --token-beam 20 \
  --nbest-size 10 \
  --lm-alpha 0.05 \
  --lm-batch-size 512 \
  --word-bonus 0.0 \
  --test-clean-output results/baseline_lm_fusion_test_clean.txt \
  --test-other-output results/baseline_lm_fusion_test_other.txt
```

### All Combined: Ablation 1 + 2 + 3 + 4

The commands below run a combined experiment with data augmentation, frozen encoder layers, maximum entropy regularization, and neural LM shallow fusion.

```bash
# Train with Ablation 1 + 2 + 3
python run/wav2vec_finetuning.py \
  --augment \
  --freeze-transformer-layers 6 \
  --max-entropy-weight 0.01 \
  --output-dir finetuning_output_all

# Inference with Ablation 4 on the combined checkpoint
python run/wav2vec_inference.py \
  --model-dir finetuning_output_all \
  --lm-model facebook/opt-125m \
  --beam-width 25 \
  --token-beam 20 \
  --nbest-size 10 \
  --lm-alpha 0.05 \
  --lm-batch-size 512 \
  --word-bonus 0.0 \
  --test-clean-output results/all_lm_test_clean.txt \
  --test-other-output results/all_lm_test_other.txt
```

### Baseline: Wav2Vec2 CTC Greedy Decoding

Train with the default fine-tuning setting, then run inference with CTC greedy decoding without an external language model.

```bash
python run/wav2vec_finetuning.py \
  --output-dir finetuning_output_baseline

python run/wav2vec_inference.py \
  --model-dir finetuning_output_baseline \
  --test-clean-output results/baseline_test_clean.txt \
  --test-other-output results/baseline_test_other.txt
```

### Ablation 1: Data Augmentation

Applies noise injection and speed perturbation to training samples. Augmentation is not applied to evaluation or inference.

```bash
python run/wav2vec_finetuning.py \
  --augment \
  --output-dir finetuning_output_aug

python run/wav2vec_inference.py \
  --model-dir finetuning_output_aug \
  --test-clean-output results/aug_test_clean.txt \
  --test-other-output results/aug_test_other.txt
```

### Ablation 2: Frozen Encoder Layers

The feature encoder is frozen by default, and this ablation additionally freezes several early Wav2Vec2 transformer encoder layers. The default experiment value is 6 layers.

```bash
python run/wav2vec_finetuning.py \
  --freeze-transformer-layers 6 \
  --output-dir finetuning_output_freeze6

python run/wav2vec_inference.py \
  --model-dir finetuning_output_freeze6 \
  --test-clean-output results/freeze6_test_clean.txt \
  --test-other-output results/freeze6_test_other.txt
```

### Ablation 3: Maximum Entropy Regularization

Adds a regularization term to the CTC loss that increases prediction entropy. It is only applied when `--max-entropy-weight` is greater than 0.

```bash
python run/wav2vec_finetuning.py \
  --max-entropy-weight 0.01 \
  --output-dir finetuning_output_maxent001

python run/wav2vec_inference.py \
  --model-dir finetuning_output_maxent001 \
  --test-clean-output results/maxent001_test_clean.txt \
  --test-other-output results/maxent001_test_other.txt
```

### Ablation 4: Neural LM Shallow Fusion

Uses Hugging Face `facebook/opt-125m` causal language model scores inside the base CTC model's prefix beam search. Each prefix decoding score combines the acoustic CTC score, neural LM log-likelihood, and word bonus.

The neural LM is not called for every character prefix. Instead, word-boundary candidates are scored in batches to keep decoding reasonably fast. Candidates without a trailing word delimiter, such as the final word, are scored one more time before the final candidate ranking at utterance end. During LM decoding, utterance-level frame progress is visible through tqdm, and `--lm-batch-size` controls the LM scoring batch size.

```text
score = ctc_score + lm_alpha * lm_score + word_bonus * word_count
```

This is shallow fusion decoding where the language model directly affects prefix pruning and final selection during beam search, not post-hoc rescoring after generating N-best candidates. This experiment changes only the decoding stage and does not retrain the model.

```bash
python run/wav2vec_inference.py \
  --model-dir finetuning_output_baseline \
  --lm-model facebook/opt-125m \
  --beam-width 25 \
  --token-beam 20 \
  --nbest-size 10 \
  --lm-alpha 0.05 \
  --lm-batch-size 512 \
  --word-bonus 0.0 \
  --test-clean-output results/lm_fusion_test_clean.txt \
  --test-other-output results/lm_fusion_test_other.txt
```

WER is computed in the same way for each result file.

```bash
python run/evaluate_wer.py results/baseline_test_clean.txt
python run/evaluate_wer.py results/baseline_test_other.txt
```
