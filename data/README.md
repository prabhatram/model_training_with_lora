# Synthetic University IT Support Training Dataset

Purpose: classroom demonstration of continued pretraining/domain adaptation followed by supervised fine-tuning (SFT).

Files:
- domain_corpus_raw.jsonl: 1,200 unlabeled domain-text records, including 4 deliberately problematic records.
- domain_corpus_clean.jsonl: cleaned corpus (1,196 records).
- sft_raw.jsonl: 720 prompt/completion examples, including 5 deliberately problematic records.
- sft_clean.jsonl: cleaned SFT data (715 records).
- problem_manifest.json: teacher key describing deliberately injected problems and fixes.
- eval.jsonl: held-out prompts for qualitative evaluation.

All data are synthetic. Credentials appearing in the deliberately problematic record are fictional, but the teaching rule is the same: secrets must never be included in training data.
