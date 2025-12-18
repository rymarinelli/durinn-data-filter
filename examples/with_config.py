from datasets import load_dataset
from dataset_risk_decorator.core import risk_guard

# Apply risk annotation with custom parameters
ds = risk_guard(
    load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train"),
    threshold=0.5,      # threshold used for is_problematic flag
    filter_mode="none", # annotate only, do not drop rows
    max_rows=2000,      # limit rows for faster iteration
)



scores = ds["risk_score"]
print("rows:", len(ds))
print("min:", min(scores))
print("mean:", sum(scores) / len(scores))
print("max:", max(scores))

# Rank samples by risk score
top_risky = ds.sort("risk_score", reverse=True).select(range(10))

for row in top_risky:
    print(row["risk_score"], row["chosen"][:200])
