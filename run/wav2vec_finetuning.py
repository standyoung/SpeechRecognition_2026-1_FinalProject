# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard imports
import argparse
import inspect
import os

# Third-party imports
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)


def parse_args():
    """Parse ablation options for Wav2Vec2 CTC fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 CTC with ablation-study options."
    )
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_DIR, "data"))
    parser.add_argument("--train-split", default="1h")
    parser.add_argument("--eval-split", default="test-clean")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--processor-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "finetuning_output"))
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--apply-spec-augment", action="store_true")
    parser.add_argument("--augment", action="store_true")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze-feature-encoder", action="store_true", dest="freeze_feature_encoder")
    freeze_group.add_argument("--unfreeze-feature-encoder", action="store_false", dest="freeze_feature_encoder")
    parser.set_defaults(freeze_feature_encoder=True)
    parser.add_argument("--freeze-transformer-layers", type=int, default=0)
    parser.add_argument("--max-entropy-weight", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate options that would otherwise fail later in Trainer."""
    if args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive.")
    if args.save_steps <= 0:
        raise ValueError("--save-steps must be positive.")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps.")
    if args.freeze_transformer_layers < 0:
        raise ValueError("--freeze-transformer-layers must be non-negative.")
    if args.max_entropy_weight < 0:
        raise ValueError("--max-entropy-weight must be non-negative.")


def build_compute_metrics(processor, wer_metric):
    """Create a WER metric callback bound to the current processor."""
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

    return compute_metrics


class MaxEntropyCTCTrainer(Trainer):
    """Trainer that can add maximum-entropy regularization to CTC loss."""

    def __init__(self, *args, max_entropy_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_entropy_weight = max_entropy_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.max_entropy_weight > 0:
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = loss - self.max_entropy_weight * entropy

        return (loss, outputs) if return_outputs else loss


def freeze_transformer_layers(model, num_layers: int) -> None:
    """Freeze the first N Wav2Vec2 encoder layers."""
    if num_layers <= 0:
        return

    encoder_layers = getattr(model.wav2vec2.encoder, "layers", [])
    for layer in encoder_layers[:num_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


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


def main():
    args = parse_args()
    validate_args(args)
    os.environ["WAV2VEC_PROCESSOR_NAME"] = args.processor_name
    os.makedirs(args.output_dir, exist_ok=True)

    # Custom imports after processor env is configured.
    import sample_util

    sample_util.MODEL_NAME = args.model_name
    processor = sample_util.processor

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_top_dir = os.path.join(args.data_dir, args.train_split)
    test_top_dir = os.path.join(args.data_dir, args.eval_split)
    train_dataset = sample_util.make_dataset(train_top_dir, augment=args.augment)
    test_dataset = sample_util.make_dataset(test_top_dir, augment=False)
    wer_metric = evaluate.load("wer")

    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding="longest"
    )

    model = AutoModelForCTC.from_pretrained(
        args.model_name,
        apply_spec_augment=args.apply_spec_augment,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        use_safetensors=True,
        vocab_size=len(processor.tokenizer)
    )
    model.config.apply_spec_augment = args.apply_spec_augment
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    freeze_transformer_layers(model, args.freeze_transformer_layers)

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "fp16": cuda_available and args.fp16,
        "dataloader_pin_memory": cuda_available,
        "per_device_eval_batch_size": args.eval_batch_size,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
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
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(processor, wer_metric),
        "max_entropy_weight": args.max_entropy_weight,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = MaxEntropyCTCTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
