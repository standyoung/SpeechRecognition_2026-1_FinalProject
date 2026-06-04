# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Seoyoung Ju(jstandzero@korea.ac.kr)"

# Standard imports
import argparse
import math
import os

# Third-party imports
import torch
from transformers import AutoModelForCausalLM, AutoModelForCTC, AutoProcessor, AutoTokenizer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)


def parse_args():
    """Parse inference and neural language model rescoring options."""
    parser = argparse.ArgumentParser(
        description="Run Wav2Vec2 inference on test-clean and test-other."
    )
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_DIR, "data"))
    parser.add_argument("--model-dir", default=os.path.join(PROJECT_DIR, "finetuning_output"))
    parser.add_argument("--processor-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--test-clean-split", default="test-clean")
    parser.add_argument("--test-other-split", default="test-other")
    parser.add_argument("--test-clean-output", default=os.path.join(PROJECT_DIR, "test_clean_result.txt"))
    parser.add_argument("--test-other-output", default=os.path.join(PROJECT_DIR, "test_other_result.txt"))
    parser.add_argument("--beam-width", type=int, default=25)
    parser.add_argument("--token-beam", type=int, default=20)
    parser.add_argument("--nbest-size", type=int, default=10)
    parser.add_argument("--rescore-model", default=None)
    parser.add_argument("--rescore-alpha", type=float, default=0.5)
    parser.add_argument("--rescore-beta", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate decoding options before loading models and datasets."""
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive.")
    if args.token_beam <= 0:
        raise ValueError("--token-beam must be positive.")
    if args.nbest_size <= 0:
        raise ValueError("--nbest-size must be positive.")


def resolve_model_dir(args):
    """Use fine-tuned model if present, otherwise fall back to processor checkpoint."""
    model_dir = os.environ.get("WAV2VEC_MODEL_DIR", args.model_dir)
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        model_dir = os.environ.get("WAV2VEC_PROCESSOR_NAME", args.processor_name)
    else:
        os.environ.setdefault("WAV2VEC_PROCESSOR_NAME", model_dir)
    return model_dir


def logadd(a: float, b: float) -> float:
    """Stable logadd for CTC prefix beam search."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class GPT2Rescorer:
    """Score CTC N-best hypotheses with a Hugging Face causal LM."""

    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def score(self, text: str) -> float:
        """Return sequence log-likelihood under the causal LM."""
        normalized = text.lower().strip()
        if not normalized:
            return -math.inf

        encoded = self.tokenizer(normalized, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        if input_ids.shape[-1] < 2:
            return 0.0

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)

        token_count = input_ids.shape[-1] - 1
        return float(-outputs.loss.item() * token_count)


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    processor,
    beam_width: int,
    token_beam: int,
    nbest_size: int
):
    """Build CTC N-best hypotheses without an external language model."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    blank_id = processor.tokenizer.pad_token_id
    special_token_ids = set(getattr(processor.tokenizer, "all_special_ids", []))
    special_token_ids.discard(blank_id)
    beams = {(): (0.0, -math.inf)}

    for frame in log_probs:
        next_beams = {}
        top_values, top_ids = torch.topk(
            frame,
            k=min(token_beam, frame.shape[-1])
        )

        for prefix, (prob_blank, prob_non_blank) in beams.items():
            prefix_score = logadd(prob_blank, prob_non_blank)

            for token_score, token_id in zip(top_values.tolist(), top_ids.tolist()):
                if token_id in special_token_ids:
                    continue

                if token_id == blank_id:
                    next_blank, next_non_blank = next_beams.get(
                        prefix,
                        (-math.inf, -math.inf)
                    )
                    next_blank = logadd(next_blank, prefix_score + token_score)
                    next_beams[prefix] = (next_blank, next_non_blank)
                    continue

                new_prefix = prefix + (token_id,)
                next_blank, next_non_blank = next_beams.get(
                    new_prefix,
                    (-math.inf, -math.inf)
                )

                if prefix and token_id == prefix[-1]:
                    same_blank, same_non_blank = next_beams.get(
                        prefix,
                        (-math.inf, -math.inf)
                    )
                    same_non_blank = logadd(
                        same_non_blank,
                        prob_non_blank + token_score
                    )
                    next_beams[prefix] = (same_blank, same_non_blank)
                    next_non_blank = logadd(
                        next_non_blank,
                        prob_blank + token_score
                    )
                else:
                    next_non_blank = logadd(
                        next_non_blank,
                        prefix_score + token_score
                    )

                next_beams[new_prefix] = (next_blank, next_non_blank)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: logadd(item[1][0], item[1][1]),
                reverse=True
            )[:beam_width]
        )

    hypotheses = []
    seen = set()
    for prefix, (prob_blank, prob_non_blank) in sorted(
        beams.items(),
        key=lambda item: logadd(item[1][0], item[1][1]),
        reverse=True
    ):
        text = processor.decode(
            list(prefix),
            group_tokens=False,
            skip_special_tokens=True
        ).strip()
        if text in seen:
            continue
        seen.add(text)
        hypotheses.append({
            "text": text,
            "acoustic_score": logadd(prob_blank, prob_non_blank)
        })
        if len(hypotheses) >= nbest_size:
            break

    return hypotheses


def rescore_hypotheses(hypotheses, rescorer, alpha: float, beta: float) -> str:
    """Select the best hypothesis using acoustic, LM, and length scores."""
    best_text = ""
    best_score = -math.inf
    for hypothesis in hypotheses:
        text = hypothesis["text"]
        lm_score = rescorer.score(text)
        length_score = len(text.split())
        total_score = (
            hypothesis["acoustic_score"]
            + alpha * lm_score
            + beta * length_score
        )
        if total_score > best_score:
            best_score = total_score
            best_text = text
    return best_text


def transcribe(
    data,
    processor,
    model,
    device,
    rescorer=None,
    beam_width=25,
    token_beam=20,
    nbest_size=10,
    rescore_alpha=0.5,
    rescore_beta=0.0
):
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

    if rescorer is not None:
        hypotheses = ctc_prefix_beam_search(
            logits[0].detach(),
            processor=processor,
            beam_width=beam_width,
            token_beam=token_beam,
            nbest_size=nbest_size
        )
        if not hypotheses:
            pred_ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(pred_ids)[0]

        return rescore_hypotheses(
            hypotheses,
            rescorer=rescorer,
            alpha=rescore_alpha,
            beta=rescore_beta
        )

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def write_results(
    dataset,
    output_file,
    processor,
    model,
    device,
    args,
    rescorer=None
):
    """Write REF/HYP pairs for a dataset."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = processor.decode(data["labels"], group_tokens=False)
            hyp = transcribe(
                data,
                processor=processor,
                model=model,
                device=device,
                rescorer=rescorer,
                beam_width=args.beam_width,
                token_beam=args.token_beam,
                nbest_size=args.nbest_size,
                rescore_alpha=args.rescore_alpha,
                rescore_beta=args.rescore_beta
            )
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")


def main():
    args = parse_args()
    validate_args(args)
    model_dir = resolve_model_dir(args)

    os.environ["WAV2VEC_PROCESSOR_NAME"] = model_dir

    # Custom imports after processor env is configured.
    import sample_util

    db_top_dir = args.data_dir
    test_clean_top_dir = os.path.join(db_top_dir, args.test_clean_split)
    test_other_top_dir = os.path.join(db_top_dir, args.test_other_split)

    test_clean_dataset = sample_util.make_dataset(test_clean_top_dir)
    test_other_dataset = sample_util.make_dataset(test_other_top_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    rescorer = None
    if args.rescore_model:
        rescorer = GPT2Rescorer(args.rescore_model, device)

    write_results(
        test_clean_dataset,
        args.test_clean_output,
        processor=processor,
        model=model,
        device=device,
        args=args,
        rescorer=rescorer
    )
    write_results(
        test_other_dataset,
        args.test_other_output,
        processor=processor,
        model=model,
        device=device,
        args=args,
        rescorer=rescorer
    )


if __name__ == "__main__":
    main()
