# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard imports
import os

# Third-party imports
import torch
from transformers import AutoModelForCTC, AutoProcessor

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)
default_model_dir = os.path.join(PROJECT_DIR, "finetuning_output")
model_dir = os.environ.get("WAV2VEC_MODEL_DIR", default_model_dir)
if not os.path.exists(os.path.join(model_dir, "config.json")):
    model_dir = os.environ.get(
        "WAV2VEC_PROCESSOR_NAME",
        "facebook/wav2vec2-base-960h"
    )
else:
    os.environ.setdefault("WAV2VEC_PROCESSOR_NAME", model_dir)

# Custom imports
import sample_util

db_top_dir = os.path.join(PROJECT_DIR, "data")
test_clean_top_dir = os.path.join(db_top_dir, "test-clean")
test_other_top_dir = os.path.join(db_top_dir, "test-other")


# TODO Complete the following parts:
test_clean_dataset = sample_util.make_dataset(test_clean_top_dir)
test_other_dataset = sample_util.make_dataset(test_other_top_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForCTC.from_pretrained(model_dir).to(device)
model.eval()
# End of TODO


def transcribe(data):
    """Transcribe one preprocessed sample without using the ASR pipeline."""
    inputs = processor(
        data["speech"],
        sampling_rate=data["sampling_rate"],
        return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(
            input_values,
            attention_mask=attention_mask
        ).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def write_results(dataset, output_file):
    """Write REF/HYP pairs for a dataset."""
    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = processor.decode(data["labels"], group_tokens=False)
            # TODO complete the following part
            hyp = transcribe(data)
            # End of TODO
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")


write_results(
    test_clean_dataset,
    os.path.join(PROJECT_DIR, "test_clean_result.txt")
)
write_results(
    test_other_dataset,
    os.path.join(PROJECT_DIR, "test_other_result.txt")
)
