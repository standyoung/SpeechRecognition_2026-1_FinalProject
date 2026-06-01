# Wav2Vec2 Fine-Tuning Notes

이 프로젝트는 LibriSpeech 기반 WebDataset shard를 사용해 Wav2Vec2 CTC 모델을 fine-tuning하고, `test-clean` 및 `test-other` 세트에서 추론 결과와 WER를 확인한다.

## 데이터 구조

현재 데이터는 프로젝트 루트의 `data/` 아래에 있다.

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

`sample_util.make_dataset()`은 `*.tar` 전체를 정렬해서 읽고, 실제 사용에 필요한 `audio`와 `text`만 가져온다.

## 전처리 변경 사항

`run/sample_util.py`에서 다음 내용을 반영했다.

- 모델/processor를 `facebook/wav2vec2-base-960h`로 통일했다.
- LibriSpeech 기준 sampling rate를 `16_000`으로 고정했다.
- raw audio bytes는 `torchaudio.load(io.BytesIO(audio))`로 읽는다.
- waveform은 CPU float tensor로 변환한 뒤 mono로 만든다.
- sampling rate가 16 kHz가 아니면 16 kHz로 resampling한다.
- transcript는 UTF-8 decode 후 공백을 정리하고 uppercase로 변환한다.
- 학습용 `input_values`, CTC label용 `labels`, 추론용 원음 배열 `speech`, `sampling_rate`를 반환한다.

## 학습 코드 변경 사항

`run/wav2vec_finetuning.py`에서 다음 내용을 반영했다.

- 데이터 경로를 실행 위치와 무관하게 프로젝트 루트 기준 `data/`로 고정했다.
- `processor`는 `sample_util.processor`를 공유해 전처리와 학습이 같은 tokenizer/feature extractor를 사용한다.
- 모델은 `sample_util.MODEL_NAME`으로 로드한다.
- label padding은 `processor.tokenizer.pad()`를 사용하고, padding 위치는 CTC loss에서 무시되도록 `-100`으로 바꾼다.
- WER metric은 한 번만 로드해서 evaluation 때 재사용한다.
- Transformers 버전에 따라 `eval_strategy` 또는 `evaluation_strategy`를 자동 선택한다.
- CPU 환경에서는 `fp16=False`, CUDA 사용 가능 시에만 `fp16=True`가 되도록 했다.
- 학습 완료 후 model과 processor를 `finetuning_output/`에 저장한다.

## 추론 코드 변경 사항

`run/wav2vec_inference.py`에서 다음 내용을 반영했다.

- 데이터 경로를 프로젝트 루트 기준 `data/`로 고정했다.
- 기본 모델 경로는 `finetuning_output/`이다.
- 아직 fine-tuned 모델이 저장되어 있지 않으면 `facebook/wav2vec2-base-960h`로 fallback해 baseline inference를 수행한다.
- pipeline에는 이미 feature-extracted `input_values`가 아니라 raw 16 kHz audio array를 넘긴다.
- 결과 파일은 프로젝트 루트에 저장된다.

```text
test_clean_result.txt
test_other_result.txt
```

## 실행 방법

프로젝트 루트로 이동한다.

```powershell
cd C:\Users\주서영대학원인공지능학과\Desktop\code\SpeechRecognition_2026-1_FinalProject
```

학습:

```powershell
python run\wav2vec_finetuning.py
```

추론:

```powershell
python run\wav2vec_inference.py
```

WER 계산:

```powershell
python run\evaluate_wer.py test_clean_result.txt
python run\evaluate_wer.py test_other_result.txt
```

## 주의 사항

- Windows에서 `python`이 Microsoft Store alias로만 잡히면 실제 Python 환경을 먼저 설치하거나 conda/venv를 활성화해야 한다.
- 첫 실행 시 Hugging Face 모델과 metric을 다운로드해야 하므로 네트워크가 필요할 수 있다.
- `test-other`에는 `shard-000005.tar`도 있으므로 `shard-000000.tar`부터 `shard-000004.tar`까지만 가정하면 일부 데이터가 누락된다.
- TODO 주석은 과제 템플릿 맥락을 유지하기 위해 코드에 남겨두었다.
