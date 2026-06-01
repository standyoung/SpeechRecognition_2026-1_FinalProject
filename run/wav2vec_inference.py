# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import pipeline

# Custom imports
import sample_util

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)

db_top_dir = os.path.join(PROJECT_DIR, "data")

test_clean_top_dir = os.path.join(db_top_dir, "test-clean")
test_other_top_dir = os.path.join(db_top_dir, "test-other")


# TODO Complete the following parts:
test_clean_dataset = sample_util.make_dataset(test_clean_top_dir)
test_other_dataset = sample_util.make_dataset(test_other_top_dir)

default_model_dir = os.path.join(PROJECT_DIR, "finetuning_output")
model_dir = os.environ.get("WAV2VEC_MODEL_DIR", default_model_dir)
if not os.path.exists(os.path.join(model_dir, "config.json")):
    model_dir = sample_util.MODEL_NAME

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    tokenizer=model_dir,
    feature_extractor=model_dir,
)
# End of TODO

# Function to write REF/HYP pairs to a file
def write_results(dataset, transcriber, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = sample_util.processor.decode(data["labels"], group_tokens=False)
            # TODO complete the following part
            hyp = transcriber(
                {
                    "array": data["speech"],
                    "sampling_rate": data["sampling_rate"],
                }
            )["text"]
            # End of TODO
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")  # double newline for readability

# Write test_clean_dataset
write_results(
    test_clean_dataset,
    transcriber,
    os.path.join(PROJECT_DIR, "test_clean_result.txt")
)

# Write test_other_dataset
write_results(
    test_other_dataset,
    transcriber,
    os.path.join(PROJECT_DIR, "test_other_result.txt")
)
