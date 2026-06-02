# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard imports
import inspect
import os

# Third-party imports
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments

# Custom imports
import sample_util

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)

# TODO: Correct paths depending on your environment
db_top_dir = os.path.join(PROJECT_DIR, "data")
train_top_dir = os.path.join(db_top_dir, "1h")
test_top_dir = os.path.join(db_top_dir, "test-clean")
processor = sample_util.processor
# End of ToDO

cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

train_dataset = sample_util.make_dataset(train_top_dir)
test_dataset = sample_util.make_dataset(test_top_dir)
wer_metric = evaluate.load("wer")


def compute_metrics(pred) -> Dict[str, float]:
    """Compute word error rate (WER) between predictions and labels."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}


@dataclass
class DataCollatorCTCWithPadding:
    """Pad input audio and CTC labels dynamically for each batch."""

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding="longest"
)

model = AutoModelForCTC.from_pretrained(
    sample_util.MODEL_NAME,
    apply_spec_augment=False,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    ignore_mismatched_sizes=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    use_safetensors=True,
    vocab_size=len(processor.tokenizer)
)
model.config.apply_spec_augment = False
model.freeze_feature_encoder()

training_args_kwargs = {
    "output_dir": os.path.join(PROJECT_DIR, "finetuning_output"),
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "warmup_steps": 100,
    "max_steps": 2000,
    "gradient_checkpointing": True,
    "fp16": cuda_available and os.environ.get("ENABLE_FP16") == "1",
    "dataloader_pin_memory": cuda_available,
    "per_device_eval_batch_size": 8,
    "save_steps": 500,
    "eval_steps": 250,
    "logging_steps": 10,
    "remove_unused_columns": False,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "push_to_hub": False,
}
strategy_arg = (
    "eval_strategy"
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters
    else "evaluation_strategy"
)
training_args_kwargs[strategy_arg] = "steps"

# TODO
# Define the training arguments for the Hugging Face Trainer.
# These control training hyperparameters and runtime behavior:
training_args = TrainingArguments(**training_args_kwargs)

# Create the Trainer instance to handle training and evaluation.
# This ties together the model, datasets, processor/data collator, and metrics.
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": test_dataset,
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
}
trainer_signature = inspect.signature(Trainer.__init__).parameters
if "processing_class" in trainer_signature:
    trainer_kwargs["processing_class"] = processor
elif "tokenizer" in trainer_signature:
    trainer_kwargs["tokenizer"] = processor.feature_extractor

trainer = Trainer(**trainer_kwargs)
# End of TODO

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
