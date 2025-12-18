from datasets import load_dataset
from dataset_risk_decorator.core import risk_guard

# Load a Hugging Face dataset split and annotate it with risk scores
ds = risk_guard(
    load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
)


scores = ds["risk_score"]
print("rows:", len(ds))
print("min:", min(scores))
print("mean:", sum(scores) / len(scores))
print("max:", max(scores))

# Sort by highest risk and inspect the top samples
top_risky = ds.sort("risk_score", reverse=True).select(range(10))

for row in top_risky:
    print(row["risk_score"], row["chosen"][:200])
