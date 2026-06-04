# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard library imports
import glob
import io
import os
import random
import re
from typing import Dict, Optional

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


def _augment_waveform(
    waveform: torch.Tensor,
    noise_std: float = 0.005,
    noise_prob: float = 0.5,
    speed_prob: float = 0.3,
    speed_factors: Optional[list] = None
) -> torch.Tensor:
    """Apply lightweight waveform augmentation for training ablations."""
    if speed_factors is None:
        speed_factors = [0.9, 1.0, 1.1]

    augmented = waveform
    if random.random() < speed_prob:
        speed = random.choice(speed_factors)
        if speed != 1.0:
            new_length = max(1, int(augmented.numel() / speed))
            augmented = torch.nn.functional.interpolate(
                augmented.view(1, 1, -1),
                size=new_length,
                mode="linear",
                align_corners=False
            ).view(-1)

    if random.random() < noise_prob and noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std

    return augmented.clamp(-1.0, 1.0)


def preprocess_sample(sample: Dict, augment: bool = False) -> Dict:
    """Preprocess one WebDataset sample for Wav2Vec2 CTC training/inference."""
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

    if augment:
        waveform = _augment_waveform(waveform)

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

    return {
        "input_values": input_values,
        "labels": labels,
        "speech": speech,
        "sampling_rate": SAMPLE_RATE,
    }


def make_dataset(data_dir: str, augment: bool = False) -> wds.WebDataset:
    """Create a WebDataset pipeline from all tar shards in data_dir."""
    shards = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No tar shards found in {data_dir}")

    dataset = (
        wds.WebDataset(shards, shardshuffle=False)
            .rename(audio="audio;wav;flac", text="text;txt;transcript")
            .to_tuple("audio", "text")
            .map(lambda sample: {"audio": sample[0], "text": sample[1]})
            .map(lambda sample: preprocess_sample(sample, augment=augment))
    )
    return dataset
