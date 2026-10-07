from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files=
        "data/domain_corpus_clean.jsonl",
    split="train"
)


print(dataset)

print(dataset[0])