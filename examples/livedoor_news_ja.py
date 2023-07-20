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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def tokenize(text: str, tokenizer=tokenizer) -> Sequence[int]:
        data = tokenizer(text, add_special_tokens=False)["input_ids"]  # type: ignore
        return np.array(data)  # type: ignore

    convert_fn = tokenize
else:
    convert_fn = identity

ds = load_dataset(
    "shunk031/livedoor-news-corpus",
    train_ratio=0.8,
    val_ratio=0.0,
    test_ratio=0.2,
    random_state=42,
    shuffle=True,
)

train_df = ds["train"].to_pandas()  # type: ignore
test_df = ds["test"].to_pandas()  # type: ignore

train_df = cast(pd.DataFrame, train_df)
test_df = cast(pd.DataFrame, test_df)

if config.debug:
    # train_df / test_df を category を考慮の上、サンプリングする
    train_df = (
        train_df.groupby("category")
        .apply(lambda x: x.sample(n=100, random_state=42))
        .reset_index(drop=True)
    )
    test_df = (
        test_df.groupby("category")
        .apply(lambda x: x.sample(n=10, random_state=42))
        .reset_index(drop=True)
    )


print(f"train: {len(train_df)}, test: {len(test_df)}")

print(train_df["category"].value_counts())
print(test_df["category"].value_counts())

X_train_text = train_df["title"] + " " + train_df["content"]
X_test_text = test_df["title"] + " " + test_df["content"]


y_train = train_df["category"].tolist()

X_train = list(map(convert_fn, X_train_text.tolist()))
X_test = list(map(convert_fn, X_test_text.tolist()))

classifier = NPCClassifier(n_jobs=-1, k=3, show_progress=True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# acc の表示

print(accuracy_score(test_df["category"].tolist(), y_pred))

# TP / FP / FN / FP の表示
from sklearn.metrics import confusion_matrix

print(confusion_matrix(test_df["category"].tolist(), y_pred))

# print(classifier._counts)
# print(classifier._scores)
# print(classifier._probabilities)

"""
0.9456890699253224
[[150   0   0   0   1   1   0   0   0]
 [  0 166   2   2   3   0   0   0   2]
 [  0   1 164   0   0   2   0   0   0]
 [  0   0   1 156   0   0   5   0   0]
 [  5   1   3   2 103   5   0   3   4]
 [  5   0   1   1   5 148   0   4   3]
 [  0   0   0   5   1   0 182   0   0]
 [  1   0   1   0   3   7   0 151   0]
 [  0   0   0   0   0   0   0   0 173]]
"""
