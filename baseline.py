import torch
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")

""" inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )


answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print(answer) """

with open(
    "data/sft_challenge_test_v2.jsonl",
    "r",
    encoding="utf-8"
) as file:

    tests = [
        json.loads(line)
        for line in file
    ]


for test in tests:

    prompt = (
        "You are a university IT "
        "support assistant.\n\n"
        "User: "
        + test["prompt"]
        + "\n\nAssistant:"
    )



    inputs = tokenizer(prompt, return_tensors="pt")


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    print("\n====================")
    print("QUESTION:")
    print(test["prompt"])

    print("\nEXPECTED POINTS:")
    print(test["expected_points"])

    print("\nMODEL:")
    print(response)