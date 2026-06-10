# Wav2Vec2 Fine-Tuning Method

이 프로젝트는 LibriSpeech 기반 WebDataset shard를 사용해 `facebook/wav2vec2-base`를 CTC 방식으로 fine-tuning하고, `test-clean` 및 `test-other`에서 추론 결과와 WER를 확인한다.

## 전체 흐름

1. `run/sample_util.py`에서 WebDataset shard를 읽고 audio/text sample을 전처리한다.
2. `run/wav2vec_finetuning.py`에서 Wav2Vec2 base checkpoint에 CTC head를 붙여 fine-tuning한다.
3. 학습된 model과 processor를 `finetuning_output/`에 저장한다.
4. `run/wav2vec_inference.py`에서 저장된 model로 `test-clean`, `test-other`를 transcribe한다.
5. `run/evaluate_wer.py`에서 REF/HYP 결과 파일의 WER를 계산한다.

## 데이터 구조

데이터는 프로젝트 루트의 `data/` 아래에 둔다.

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

각 tar shard 내부 sample은 다음 확장자를 가진다.

```text
<sample-id>.audio
<sample-id>.text
<sample-id>.meta
```

코드에서는 `*.tar` 전체를 정렬해서 읽고, 학습과 평가에 필요한 `audio`, `text`만 사용한다. WebDataset은 `shardshuffle=False`로 설정해 shard 순서를 고정한다.

Fine-tuning 중에는 `test-clean`, `test-other`를 validation으로 사용하지 않는다. 두 split은 최종 inference 및 WER 계산에만 사용한다. 모델 선택을 위한 validation set은 `data/1h` 내부 shard에서 분리한다.

기본 설정(`--val-ratio 0.2`)에서는 `data/1h`의 5개 shard 중 마지막 1개 shard를 validation으로 사용한다.

```text
data/1h/
  shard-000000.tar  -> train
  shard-000001.tar  -> train
  shard-000002.tar  -> train
  shard-000003.tar  -> train
  shard-000004.tar  -> validation
```

validation split은 학습 중 `eval_steps`마다 WER를 계산하고 best checkpoint를 선택하는 용도로만 사용한다. 학습 augmentation을 켜더라도 validation에는 augmentation을 적용하지 않는다.

## 전처리 방법

`run/sample_util.py`는 한 sample을 다음 방식으로 변환한다.

- audio bytes를 `soundfile`로 먼저 decode하고, 실패하면 `torchaudio.load()`를 사용한다.
- waveform을 CPU float tensor로 변환한다.
- stereo 또는 multi-channel audio는 mono로 변환한다.
- sampling rate가 16 kHz가 아니면 `torchaudio.functional.resample()`로 16 kHz로 변환한다.
- transcript는 UTF-8 text로 읽고, 공백을 정리한 뒤 uppercase로 맞춘다.
- processor feature extractor로 audio를 `input_values`로 변환한다.
- processor tokenizer로 transcript를 CTC label id인 `labels`로 변환한다.
- inference를 위해 raw 16 kHz numpy array인 `speech`와 `sampling_rate`도 함께 반환한다.

현재 코드의 checkpoint 설정은 다음과 같다.

```text
MODEL_NAME = facebook/wav2vec2-base
PROCESSOR_NAME = facebook/wav2vec2-base-960h
```

`facebook/wav2vec2-base`는 pretraining checkpoint라 ASR tokenizer를 직접 제공하지 않을 수 있다. 따라서 acoustic model은 base checkpoint를 사용하고, CTC label 생성과 decoding에는 ASR tokenizer가 포함된 `facebook/wav2vec2-base-960h` processor를 사용한다.

## Fine-Tuning 방법

`run/wav2vec_finetuning.py`는 `AutoModelForCTC.from_pretrained()`로 `facebook/wav2vec2-base`를 CTC 모델로 로드한다. base checkpoint는 ASR용 CTC head가 없는 pretraining model이므로 `lm_head`는 새로 초기화된다. 반대로 pretraining에 사용된 quantizer/projection weight는 CTC 학습에 사용되지 않는다.

모델 로드 시 핵심 설정은 다음과 같다.

```text
apply_spec_augment = False
ctc_zero_infinity = True
ignore_mismatched_sizes = True
use_safetensors = True
vocab_size = len(processor.tokenizer)
```

`apply_spec_augment=False`는 train mode에서 NaN loss/gradient가 발생하는 것을 피하기 위한 안정화 설정이다. `ctc_zero_infinity=True`는 CTC loss 계산 중 무한대 loss가 gradient를 망가뜨리는 것을 방지한다. `use_safetensors=True`는 현재 torch 환경에서 `facebook/wav2vec2-base`를 안전하게 로드하기 위해 사용한다.

## Batch 구성과 CTC Label Padding

Wav2Vec2 CTC 학습에서는 audio 길이와 transcript 길이가 sample마다 다르므로 dynamic padding을 사용한다.

`DataCollatorCTCWithPadding`은 다음 작업을 한다.

- `processor.pad()`로 batch 내 audio `input_values`를 가장 긴 audio 길이에 맞춰 padding한다.
- `processor.tokenizer.pad()`로 label token id를 가장 긴 label 길이에 맞춰 padding한다.
- label padding 위치를 `-100`으로 바꾼다.

label padding을 `-100`으로 바꾸는 이유는 PyTorch CTC loss가 해당 위치를 loss 계산에서 무시하도록 하기 위해서다. 평가 시에는 `-100`을 다시 tokenizer의 pad token id로 되돌린 뒤 text로 decode한다.

## 학습 설정

A5000 24GB GPU 기준 기본 학습 설정은 다음과 같다.

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

feature encoder는 `model.freeze_feature_encoder()`로 freeze한다. 1h fine-tuning처럼 데이터가 작은 경우 low-level acoustic feature extractor를 고정하면 학습 안정성이 좋아진다.

fp16은 기본적으로 끄고 fp32 + TF32로 학습한다. fp16을 실험하려면 `--fp16` argument를 사용한다.

## Evaluation 방법

fine-tuning 중에는 `data/1h`에서 분리한 validation shard를 eval dataset으로 사용한다. 기본 설정에서는 `shard-000004.tar`가 validation set이다. 모델 output logits에서 `argmax`로 token id를 고르고, processor로 text를 decode한다. reference label은 `-100`을 pad token id로 복원한 뒤 decode한다. 이후 `evaluate.load("wer")`로 WER를 계산한다.

best checkpoint는 validation WER가 낮은 모델을 기준으로 선택한다.

```text
metric_for_best_model = wer
greater_is_better = False
```

`test-clean`과 `test-other`는 fine-tuning 중 eval dataset이나 checkpoint 선택에 사용하지 않고, 학습이 끝난 뒤 최종 test WER를 계산할 때만 사용한다.

## Inference 방법

`run/wav2vec_inference.py`는 기본적으로 `finetuning_output/`에 저장된 fine-tuned model을 사용한다. 아직 학습된 모델이 없으면 baseline inference를 위해 `PROCESSOR_NAME` checkpoint로 fallback한다.

기본 추론은 외부 language model 없이 CTC greedy decoding을 사용한다. `--lm-model`을 지정하면 CTC prefix beam search 안에서 Hugging Face causal language model 점수를 함께 사용하는 neural LM shallow fusion decoding을 수행한다.

Neural LM shallow fusion은 beam search 중 후보 점수에 LM log-likelihood를 직접 더해 prefix pruning과 최종 선택에 반영한다. 이전처럼 모든 character prefix마다 OPT를 호출하면 `frame 수 * beam width * token beam` 규모로 LM forward가 발생해 매우 느려진다. 현재 구현은 속도를 위해 단어 경계 후보가 생길 때만 LM 점수를 batch로 갱신하고, 각 utterance 마지막 후보는 최종 선택 전에 한 번 더 scoring한다.

`--lm-model`을 사용할 때는 dataset 단위 tqdm 아래에 utterance 내부 CTC frame 진행률도 표시된다. 내부 진행바에는 현재 beam 수, 이번 frame에서 LM scoring 대상이 된 prefix 수, LM cache 크기가 함께 표시된다. 진행바가 너무 많으면 `--no-lm-progress`로 끌 수 있다.

```text
score = ctc_score + lm_alpha * lm_score + word_bonus * word_count
```

shallow fusion을 사용할 때 prefix text는 이미 CTC prefix beam search가 collapse한 token sequence이므로, decoding 시 `group_tokens=False`를 사용해 반복 문자(`LL`, `EE` 등)가 다시 collapse되지 않게 한다.

속도가 더 필요하면 `--beam-width`와 `--token-beam`을 줄인다. 두 값이 클수록 acoustic 후보가 많아지고, 단어 경계에서 batch scoring해야 하는 LM 후보도 늘어난다. `--lm-batch-size`는 neural LM scoring의 최대 batch 크기이며 기본값은 `512`다. GPU memory가 충분한데 사용률이 낮으면 값을 키워볼 수 있다.

결과는 프로젝트 루트에 저장된다.

```text
test_clean_result.txt
test_other_result.txt
```

각 파일은 다음 형식을 따른다.

```text
REF: reference transcript
HYP: predicted transcript
```

## 실행 방법

프로젝트 루트로 이동하고 conda 환경을 활성화한다.

```bash
cd ~/disk2/syju/code/SpeechRecognition_2026-1_FinalProject
conda activate syju_speech
```

학습:

```bash
python run/wav2vec_finetuning.py
```

추론:

```bash
python run/wav2vec_inference.py
```

WER 계산:
`run/evaluate_wer.py`는 REF/HYP 파일을 읽어 reference list와 hypothesis list를 만든 뒤 `jiwer.wer()`로 WER를 계산한다.

```bash
python run/evaluate_wer.py test_clean_result.txt
python run/evaluate_wer.py test_other_result.txt
```

## 참고 사항

- `facebook/wav2vec2-base`를 CTC 모델로 로드하면 `lm_head`가 `MISSING`으로 표시된다. 이는 CTC output head가 새로 초기화된다는 뜻이며 fine-tuning 대상이다.
- pretraining용 quantizer/projection weight가 `UNEXPECTED`로 표시되는 것은 CTC 학습에서 사용하지 않는 weight가 checkpoint에 있기 때문이다.
- `loss=0`, `grad_norm=nan`이 같이 나오면 정상 학습이 아니므로 run을 중단하고 기본 fp32 설정과 `apply_spec_augment=False` 설정을 확인한다.
- ablation study를 위해 주요 실험 옵션은 command line argument로 제어한다.

## 현재 코드 변경 요약

- Fine-tuning 중 `test-clean`, `test-other`를 eval dataset으로 사용하지 않는다.
- `data/1h` shard를 train/validation으로 분리하고, validation WER 기준으로 best checkpoint를 선택한다.
- 기본 `--val-ratio 0.2`에서는 `shard-000000.tar`부터 `shard-000003.tar`까지 train, `shard-000004.tar`를 validation으로 사용한다.
- 기존 N-best rescoring 방식은 neural LM shallow fusion으로 대체했다.
- Shallow fusion은 `--lm-model`, `--lm-alpha`, `--word-bonus` 옵션으로 제어한다.
- Neural LM score는 모든 character prefix가 아니라 단어 경계 후보에서 batch로 계산해 inference 속도를 개선한다.
- LM decoding 중 utterance 내부 frame 진행률을 tqdm으로 표시하고, `--lm-batch-size`로 LM scoring batch 크기를 조절할 수 있다.
- 기존 WER 수치는 split과 decoding 방식 변경 전 결과이므로 재실험이 필요하다.

## Baseline 요약

`run/` 폴더의 현재 baseline은 `facebook/wav2vec2-base`에서 시작해 LibriSpeech 1h shard로 CTC fine-tuning을 수행하는 구성이다.

- `facebook/wav2vec2-base` checkpoint를 acoustic model 초기값으로 사용한다.
- grapheme 기반 tokenizer/processor를 사용해 transcript를 CTC label로 변환한다.
- `AutoModelForCTC`로 CTC head를 붙이고 fine-tuning한다.
- fine-tuned model과 processor는 `finetuning_output/`에 저장한다.
- 저장된 모델로 `test-clean`, `test-other`를 추론해 `test_clean_result.txt`, `test_other_result.txt`를 생성한다.
- `run/evaluate_wer.py`로 REF/HYP 결과 파일의 WER를 계산한다.

## Baseline 결과

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

## Ablation Study 계획

다음 실험은 같은 train/validation split을 사용하고, 학습 완료 후 `test-clean`과 `test-other` WER를 비교한다. 아무 ablation option도 주지 않고 `python run/wav2vec_finetuning.py`를 실행하면 기본 설정이 baseline이다. 결과 파일명과 output directory를 분리해두면 실험별 결과를 쉽게 업데이트할 수 있다.

| Experiment | Training option | Decoding option | test-clean WER | test-other WER | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Baseline | default CTC fine-tuning | greedy decoding | 0.2800 | 0.4035 | `wav2vec2 CTC greedy decoding` |
| Ablation 1 | `--augment` | greedy decoding | 0.2434 | 0.3417 | data augmentation |
| Ablation 2 | `--freeze-transformer-layers 6` | greedy decoding | 0.2439 | 0.3219 | frozen encoder layers |
| Ablation 3 | `--max-entropy-weight 0.01` | greedy decoding | 0.2553 | 0.3667 | maximum entropy regularization |
| Ablation 4 | default CTC fine-tuning | `--lm-model facebook/opt-125m` | TBD | TBD | neural LM shallow fusion |
| Ablation 1+2+3 | `--augment` + `--freeze-transformer-layers 6` + `--max-entropy-weight 0.01` | greedy decoding | 0.2316 | 0.3005 | combined training |
| Ablation 1+2+3+4 | `--augment` + `--freeze-transformer-layers 6` + `--max-entropy-weight 0.01` | `--lm-model facebook/opt-125m` | TBD | TBD | combined training + decoding setting |

### Ablation 학습 명령어

Baseline은 추가 option 없이 학습한다. 각 ablation은 output directory를 다르게 저장해 checkpoint와 결과를 분리한다.

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

Neural LM shallow fusion은 학습 ablation이 아니라 decoding ablation이므로, baseline checkpoint를 사용해 inference 단계에서 적용한다.

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

### All combined: Ablation 1 + 2 + 3 + 4

아래 명령은 data augmentation, frozen encoder layers, maximum entropy regularization, neural LM shallow fusion을 모두 함께 적용하는 조합 실험이다.

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
  --test-clean-output results/all_test_clean.txt \
  --test-other-output results/all_test_other.txt
```

### Baseline: wav2vec2 CTC greedy decoding

기본 fine-tuning 설정으로 학습한 뒤, 외부 language model 없이 CTC greedy decoding으로 추론한다.

```bash
python run/wav2vec_finetuning.py \
  --output-dir finetuning_output_baseline

python run/wav2vec_inference.py \
  --model-dir finetuning_output_baseline \
  --test-clean-output results/baseline_test_clean.txt \
  --test-other-output results/baseline_test_other.txt
```

### Ablation 1: data augmentation

학습 sample에 noise injection과 speed perturbation을 적용한다. Evaluation과 inference에는 augmentation을 적용하지 않는다.

```bash
python run/wav2vec_finetuning.py \
  --augment \
  --output-dir finetuning_output_aug

python run/wav2vec_inference.py \
  --model-dir finetuning_output_aug \
  --test-clean-output results/aug_test_clean.txt \
  --test-other-output results/aug_test_other.txt
```

### Ablation 2: frozen encoder layers

feature encoder는 기본적으로 freeze하고, 추가로 Wav2Vec2 transformer encoder 앞쪽 layer 일부를 freeze한다. 기본 실험값은 6개 layer로 둔다.

```bash
python run/wav2vec_finetuning.py \
  --freeze-transformer-layers 6 \
  --output-dir finetuning_output_freeze6

python run/wav2vec_inference.py \
  --model-dir finetuning_output_freeze6 \
  --test-clean-output results/freeze6_test_clean.txt \
  --test-other-output results/freeze6_test_other.txt
```

### Ablation 3: maximum entropy regularization

CTC loss에 prediction entropy를 높이는 regularization term을 추가한다. `--max-entropy-weight`가 0보다 클 때만 적용된다.

```bash
python run/wav2vec_finetuning.py \
  --max-entropy-weight 0.01 \
  --output-dir finetuning_output_maxent001

python run/wav2vec_inference.py \
  --model-dir finetuning_output_maxent001 \
  --test-clean-output results/maxent001_test_clean.txt \
  --test-other-output results/maxent001_test_other.txt
```

### Ablation 4: neural LM shallow fusion

기본 CTC 모델의 prefix beam search 안에서 Hugging Face `facebook/opt-125m` causal language model 점수를 함께 사용한다. 각 prefix의 decoding score는 acoustic CTC score, neural LM log-likelihood, word bonus를 결합한다.

Neural LM은 매 character prefix마다 호출하지 않고, 단어 경계 후보를 batch로 scoring해 decoding 속도를 유지한다. 마지막 단어처럼 아직 trailing word delimiter가 붙지 않은 후보는 utterance 종료 후 최종 후보 ranking 전에 한 번 더 LM scoring한다. LM decoding 중에는 utterance 내부 frame 진행률을 tqdm으로 확인할 수 있고, `--lm-batch-size`로 LM scoring batch 크기를 조절할 수 있다.

```text
score = ctc_score + lm_alpha * lm_score + word_bonus * word_count
```

이 방식은 N-best 후보를 만든 뒤 후처리로 재정렬하는 rescoring이 아니라, beam search 중 prefix pruning과 최종 선택에 language model이 직접 관여하는 shallow fusion decoding이다. 이 실험은 모델 학습을 새로 하지 않고 decoding 단계만 바꾼다.

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

WER 계산은 각 결과 파일에 대해 동일하게 수행한다.

```bash
python run/evaluate_wer.py results/baseline_test_clean.txt
python run/evaluate_wer.py results/baseline_test_other.txt
```
