# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
import re
from typing import Dict

# Third-party imports
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
SAMPLE_RATE = 16000
_WHITESPACE_RE = re.compile(r"\s+")

def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    # TODO Implement this function
    audio = sample["audio"]
    text = sample["text"]

    if isinstance(audio, tuple):
        waveform, sampling_rate = audio
    else:
        waveform, sampling_rate = torchaudio.load(io.BytesIO(audio))

    waveform = waveform.float()
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

    speech = waveform.numpy()
    input_values = processor(
        speech,
        sampling_rate=SAMPLE_RATE
    ).input_values[0]
    labels = processor(text=transcript).input_ids
    # End of TODO
    return {"input_values": input_values, "labels": labels}


def make_dataset(data_dir: str) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    dataset = (
        wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
            .decode(wds.torch_audio)
            .to_tuple("audio", "text", "meta")
            .map(lambda sample: {"audio": sample[0], "text": sample[1], "meta": sample[2]})
            .map(preprocess_sample)
    )
    return dataset
