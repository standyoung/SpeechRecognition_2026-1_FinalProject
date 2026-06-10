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
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForCTC, AutoProcessor, AutoTokenizer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RUN_DIR)


def parse_args():
    """Parse inference and neural language model decoding options."""
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
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--lm-alpha", type=float, default=0.05)
    parser.add_argument("--word-bonus", type=float, default=0.0)
    parser.add_argument(
        "--rescore-model",
        dest="lm_model",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rescore-alpha",
        dest="lm_alpha",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rescore-beta",
        dest="word_bonus",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS
    )
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate decoding options before loading models and datasets."""
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive.")
    if args.token_beam <= 0:
        raise ValueError("--token-beam must be positive.")
    if args.nbest_size <= 0:
        raise ValueError("--nbest-size must be positive.")
    if args.lm_alpha < 0:
        raise ValueError("--lm-alpha must be non-negative.")


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


class CausalLMScorer:
    """Score partial CTC hypotheses with a Hugging Face causal LM."""

    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        lm_dtype = torch.float16 if device.type == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            torch_dtype=lm_dtype,
        ).to(device)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = device
        self.use_amp = device.type == "cuda"
        self.cache = {"": 0.0}

    def score(self, text: str) -> float:
        """Return sequence log-likelihood under the causal LM."""
        return self.score_many([text])[0]

    def score_many(self, texts):
        """Return sequence log-likelihoods under the causal LM."""
        normalized_texts = [text.lower().strip() for text in texts]
        scores = [None] * len(normalized_texts)
        pending_texts = []

        for idx, normalized in enumerate(normalized_texts):
            if normalized in self.cache:
                scores[idx] = self.cache[normalized]
            elif not normalized:
                self.cache[normalized] = 0.0
                scores[idx] = 0.0
            elif normalized not in pending_texts:
                pending_texts.append(normalized)

        if pending_texts:
            encoded = self.tokenizer(
                pending_texts,
                return_tensors="pt",
                padding=True
            )
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).logits
                else:
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).logits

            token_counts = attention_mask[:, 1:].sum(dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            sequence_scores = (
                token_log_probs * attention_mask[:, 1:]
            ).sum(dim=-1)

            for text, token_count, score in zip(
                pending_texts,
                token_counts.tolist(),
                sequence_scores.tolist()
            ):
                self.cache[text] = float(score) if token_count > 0 else 0.0

        for idx, normalized in enumerate(normalized_texts):
            if scores[idx] is None:
                scores[idx] = self.cache[normalized]

        return scores


def combined_score(
    beam_state,
    lm_alpha: float = 0.0,
    word_bonus: float = 0.0
) -> float:
    """Combine CTC, neural LM, and word bonus scores for beam ranking."""
    prob_blank, prob_non_blank, lm_score, word_count = beam_state
    return (
        logadd(prob_blank, prob_non_blank)
        + lm_alpha * lm_score
        + word_bonus * word_count
    )


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    processor,
    beam_width: int,
    token_beam: int,
    nbest_size: int,
    lm_scorer=None,
    lm_alpha: float = 0.0,
    word_bonus: float = 0.0
):
    """Build CTC N-best hypotheses with optional neural LM shallow fusion."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    blank_id = processor.tokenizer.pad_token_id
    special_token_ids = set(getattr(processor.tokenizer, "all_special_ids", []))
    special_token_ids.discard(blank_id)
    word_delimiter_id = getattr(processor.tokenizer, "word_delimiter_token_id", None)
    if word_delimiter_id is not None:
        special_token_ids.discard(word_delimiter_id)
    text_cache = {(): ""}
    beams = {(): (0.0, -math.inf, 0.0, 0)}

    def decode_prefix(prefix):
        if prefix not in text_cache:
            text = processor.batch_decode(
                [list(prefix)],
                group_tokens=False
            )[0].strip()
            text_cache[prefix] = text
        return text_cache[prefix]

    def apply_lm_scores(next_beams, prefixes):
        if lm_scorer is None or not prefixes:
            return
        scored_prefixes = [prefix for prefix in prefixes if prefix in next_beams]
        texts = [decode_prefix(prefix) for prefix in scored_prefixes]
        lm_scores = lm_scorer.score_many(texts)
        for prefix, text, lm_score in zip(scored_prefixes, texts, lm_scores):
            prob_blank, prob_non_blank, _, _ = next_beams[prefix]
            next_beams[prefix] = (
                prob_blank,
                prob_non_blank,
                lm_score,
                len(text.split())
            )

    for frame in log_probs:
        next_beams = {}
        pending_lm_prefixes = set()
        top_values, top_ids = torch.topk(
            frame,
            k=min(token_beam, frame.shape[-1])
        )

        for prefix, (prob_blank, prob_non_blank, lm_score, word_count) in beams.items():
            prefix_score = logadd(prob_blank, prob_non_blank)

            for token_score, token_id in zip(top_values.tolist(), top_ids.tolist()):
                if token_id in special_token_ids:
                    continue

                if token_id == blank_id:
                    next_blank, next_non_blank, _, _ = next_beams.get(
                        prefix,
                        (-math.inf, -math.inf, lm_score, word_count)
                    )
                    next_blank = logadd(next_blank, prefix_score + token_score)
                    next_beams[prefix] = (
                        next_blank,
                        next_non_blank,
                        lm_score,
                        word_count
                    )
                    continue

                new_prefix = prefix + (token_id,)
                new_lm_score = lm_score
                new_word_count = word_count
                if token_id == word_delimiter_id:
                    pending_lm_prefixes.add(new_prefix)
                next_blank, next_non_blank, _, _ = next_beams.get(
                    new_prefix,
                    (-math.inf, -math.inf, new_lm_score, new_word_count)
                )

                if prefix and token_id == prefix[-1]:
                    same_blank, same_non_blank, _, _ = next_beams.get(
                        prefix,
                        (-math.inf, -math.inf, lm_score, word_count)
                    )
                    same_non_blank = logadd(
                        same_non_blank,
                        prob_non_blank + token_score
                    )
                    next_beams[prefix] = (
                        same_blank,
                        same_non_blank,
                        lm_score,
                        word_count
                    )
                    next_non_blank = logadd(
                        next_non_blank,
                        prob_blank + token_score
                    )
                else:
                    next_non_blank = logadd(
                        next_non_blank,
                        prefix_score + token_score
                    )

                next_beams[new_prefix] = (
                    next_blank,
                    next_non_blank,
                    new_lm_score,
                    new_word_count
                )

        apply_lm_scores(next_beams, pending_lm_prefixes)
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: combined_score(
                    item[1],
                    lm_alpha=lm_alpha,
                    word_bonus=word_bonus
                ),
                reverse=True
            )[:beam_width]
        )

    final_items = []
    final_prefixes = []
    final_texts = []
    for prefix in beams:
        final_prefixes.append(prefix)
        final_texts.append(decode_prefix(prefix))

    final_lm_scores = (
        lm_scorer.score_many(final_texts)
        if lm_scorer is not None
        else [beam_state[2] for beam_state in beams.values()]
    )
    for prefix, beam_state, text, lm_score in zip(
        final_prefixes,
        beams.values(),
        final_texts,
        final_lm_scores
    ):
        prob_blank, prob_non_blank, _, _ = beam_state
        final_state = (
            prob_blank,
            prob_non_blank,
            lm_score,
            len(text.split())
        )
        final_items.append((prefix, text, final_state))

    hypotheses = []
    seen = set()
    for _, text, beam_state in sorted(
        final_items,
        key=lambda item: combined_score(
            item[2],
            lm_alpha=lm_alpha,
            word_bonus=word_bonus
        ),
        reverse=True
    ):
        prob_blank, prob_non_blank, lm_score, word_count = beam_state
        if text in seen:
            continue
        seen.add(text)
        hypotheses.append({
            "text": text,
            "acoustic_score": logadd(prob_blank, prob_non_blank),
            "lm_score": lm_score,
            "word_count": word_count,
            "score": combined_score(
                beam_state,
                lm_alpha=lm_alpha,
                word_bonus=word_bonus
            )
        })
        if len(hypotheses) >= nbest_size:
            break

    return hypotheses


def transcribe(
    data,
    processor,
    model,
    device,
    lm_scorer=None,
    beam_width=25,
    token_beam=20,
    nbest_size=10,
    lm_alpha=0.05,
    word_bonus=0.0
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

    if lm_scorer is not None:
        hypotheses = ctc_prefix_beam_search(
            logits[0].detach(),
            processor=processor,
            beam_width=beam_width,
            token_beam=token_beam,
            nbest_size=nbest_size,
            lm_scorer=lm_scorer,
            lm_alpha=lm_alpha,
            word_bonus=word_bonus
        )
        if not hypotheses:
            pred_ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(pred_ids)[0]

        return hypotheses[0]["text"]

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def write_results(
    dataset,
    output_file,
    processor,
    model,
    device,
    args,
    lm_scorer=None,
    split_name="dataset"
):
    """Write REF/HYP pairs for a dataset."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_samples = len(dataset) if hasattr(dataset, "__len__") else None
    progress_desc = f"Decoding {split_name}"

    with open(output_file, "w", encoding="utf-8") as f:
        for data in tqdm(dataset, total=total_samples, desc=progress_desc, unit="utt"):
            ref = processor.decode(data["labels"], group_tokens=False)
            hyp = transcribe(
                data,
                processor=processor,
                model=model,
                device=device,
                lm_scorer=lm_scorer,
                beam_width=args.beam_width,
                token_beam=args.token_beam,
                nbest_size=args.nbest_size,
                lm_alpha=args.lm_alpha,
                word_bonus=args.word_bonus
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
    lm_scorer = None
    if args.lm_model:
        lm_scorer = CausalLMScorer(args.lm_model, device)

    write_results(
        test_clean_dataset,
        args.test_clean_output,
        processor=processor,
        model=model,
        device=device,
        args=args,
        lm_scorer=lm_scorer,
        split_name=args.test_clean_split
    )
    write_results(
        test_other_dataset,
        args.test_other_output,
        processor=processor,
        model=model,
        device=device,
        args=args,
        lm_scorer=lm_scorer,
        split_name=args.test_other_split
    )


if __name__ == "__main__":
    main()
