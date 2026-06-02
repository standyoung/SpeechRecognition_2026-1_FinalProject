import argparse
from jiwer import wer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Compute WER from a REF/HYP text file")
parser.add_argument("file_path", type=str, help="Path to the REF/HYP results file")
args = parser.parse_args()

refs = []
hyps = []

with open(args.file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove empty lines

for i in range(0, len(lines), 2):  # every two lines: REF and HYP
    ref_line = lines[i]
    hyp_line = lines[i + 1]

    # Extract text after "REF:" / "HYP:"
    ref_text = ref_line[len("REF:"):].strip()
    hyp_text = hyp_line[len("HYP:"):].strip()

    refs.append(ref_text)
    hyps.append(hyp_text)

# Compute WER
test_wer = wer(refs, hyps)
print(f"WER: {test_wer:.4f}")

