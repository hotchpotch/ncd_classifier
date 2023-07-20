import argparse
from typing import Sequence, cast
from datasets import load_dataset
import numpy as np
import pandas as pd
from npc_classifier.npc_classifier import NPCClassifier
from sklearn.metrics import accuracy_score

args = argparse.ArgumentParser()
args.add_argument("--debug", action="store_true")
args.add_argument("--use_tokenizer", action="store_true")
config = args.parse_args()

identity = lambda x: x

if config.use_tokenizer:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def tokenize(text: str, tokenizer=tokenizer) -> Sequence[int]:
        data = tokenizer(text, add_special_tokens=False)["input_ids"]  # type: ignore
        return np.array(data)  # type: ignore

    convert_fn = tokenize
else:
    convert_fn = identity

ds = load_dataset(
    "ag_news",
)

train_df = ds["train"].to_pandas()  # type: ignore
test_df = ds["test"].to_pandas()  # type: ignore

train_df = cast(pd.DataFrame, train_df)
test_df = cast(pd.DataFrame, test_df)

if config.debug:
    # train_df / test_df を label を考慮の上、サンプリングする
    train_df = (
        train_df.groupby("label")
        .apply(lambda x: x.sample(n=10000, random_state=42))
        .reset_index(drop=True)
    )
    test_df = (
        test_df.groupby("label")
        .apply(lambda x: x.sample(n=100, random_state=42))
        .reset_index(drop=True)
    )


print(f"train: {len(train_df)}, test: {len(test_df)}")

print(train_df["label"].value_counts())
print(test_df["label"].value_counts())

X_train_text = train_df["text"]
X_test_text = test_df["text"]


y_train = train_df["label"].tolist()

X_train = list(map(convert_fn, X_train_text.tolist()))
X_test = list(map(convert_fn, X_test_text.tolist()))

classifier = NPCClassifier(n_jobs=-1, k=5, show_progress=True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# acc の表示

print(accuracy_score(test_df["label"].tolist(), y_pred))

# TP / FP / FN / FP の表示
from sklearn.metrics import confusion_matrix

print(confusion_matrix(test_df["label"].tolist(), y_pred))

# print(classifier._counts)
# print(classifier._scores)
# print(classifier._probabilities)

"""
0.8976315789473684
[[1718   47   83   52]
 [  20 1838   23   19]
 [  72   31 1635  162]
 [  81   37  151 1631]]
"""
