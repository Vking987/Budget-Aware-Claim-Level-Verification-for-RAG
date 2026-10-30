from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="train[:1]"
)

example = dataset[0]

print("TYPE:")
print(type(example["supporting_facts"]))

print("\nSUPPORTING FACTS:")
print(example["supporting_facts"])

print("\nTYPE OF EACH ELEMENT:")
for x in example["supporting_facts"]:
    print(type(x), repr(x))