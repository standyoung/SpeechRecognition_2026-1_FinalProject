# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard library imports
import glob
import io
import os
import re
from typing import Dict

# Third-party imports
import torch
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

MODEL_NAME = "facebook/wav2vec2-base"
PROCESSOR_NAME = os.environ.get("WAV2VEC_PROCESSOR_NAME", "facebook/wav2vec2-base-960h")
SAMPLE_RATE = 16_000

processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess one WebDataset sample for Wav2Vec2 CTC training/inference."""
    # TODO Implement this function
    audio = sample["audio"]
    text = sample["text"]

    if isinstance(audio, tuple):
        waveform, sampling_rate = audio
    else:
        try:
            import soundfile as sf
            audio_array, sampling_rate = sf.read(
                io.BytesIO(audio),
                dtype="float32",
                always_2d=True
            )
            waveform = torch.from_numpy(audio_array).transpose(0, 1)
        except (ImportError, OSError, RuntimeError):
            waveform, sampling_rate = torchaudio.load(io.BytesIO(audio))

    waveform = waveform.detach().cpu().float()
    if waveform.ndim > 1:
        waveform = waveform.squeeze(0) if waveform.shape[0] == 1 else waveform.mean(dim=0)

    if sampling_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sampling_rate,
            new_freq=SAMPLE_RATE
        )

    if isinstance(text, bytes):
        transcript = text.decode("utf-8").strip()
    else:
        transcript = str(text).strip()
    transcript = _WHITESPACE_RE.sub(" ", transcript).upper()

    speech = waveform.contiguous().numpy()
    input_values = processor(
        speech,
        sampling_rate=SAMPLE_RATE
    ).input_values[0]
    labels = processor(text=transcript).input_ids
    # End of TODO

    return {
        "input_values": input_values,
        "labels": labels,
        "speech": speech,
        "sampling_rate": SAMPLE_RATE,
    }


def make_dataset(data_dir: str) -> wds.WebDataset:
    """Create a WebDataset pipeline from all tar shards in data_dir."""
    shards = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No tar shards found in {data_dir}")

    dataset = (
        wds.WebDataset(shards, shardshuffle=False)
            .rename(audio="audio;wav;flac", text="text;txt;transcript")
            .to_tuple("audio", "text")
            .map(lambda sample: {"audio": sample[0], "text": sample[1]})
            .map(preprocess_sample)
    )
    return dataset
