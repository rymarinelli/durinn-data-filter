from datasets import load_dataset
from dataset_risk_decorator.core import risk_guard

@risk_guard(
    threshold=0.5,
    filter_mode="none",
    max_rows=2000,
)
def load_data():
    return load_dataset(
        "CyberNative/Code_Vulnerability_Security_DPO"
    )

ds = load_data()
train = ds["train"]

top_risky = train.sort("risk_score", reverse=True).select(range(10))

for row in top_risky:
    print(row["risk_score"], row["chosen"][:200])
