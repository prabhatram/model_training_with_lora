import json
from collections import Counter


FILE = "data/domain_corpus_raw.jsonl"


records = []

with open(FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        records.append(record)

print("Number of records:", len(records))

topics = Counter(record["topic"] for record in records)

print("\nTopics:")

for topic, count in topics.items():
    print(topic, count)


print("\nSample records:")
for record in records[:5]:
    print("\n--------------------")
    print(record)