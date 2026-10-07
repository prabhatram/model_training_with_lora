Improved SFT package

Fix 1: More diverse scenario-based examples.
Fix 2: Whole semantic scenarios are assigned to train OR validation OR test; paraphrases do not cross splits.
Fix 3: sft_test.jsonl and sft_challenge_test.jsonl are untouched final tests. Never pass them to SFTTrainer.

Rerun:
Do NOT rerun 04_domain_train.py if models/domain_adapter is intact. The domain-adaptation data was not the source of the SFT leakage.
Run 06_sft_lora_v2.py from the existing models/domain_adapter.
It writes a fresh models/specialized_adapter_v2.
Then evaluate specialized_adapter_v2 on sft_test.jsonl and sft_challenge_test.jsonl.

Recommended first rerun: 1 epoch, learning rate 5e-5.
