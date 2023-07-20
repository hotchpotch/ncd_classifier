import argparse
from typing import Sequence, cast
from datasets import load_dataset
import numpy as np
import pandas as pd
from ncd_classifier.ncd_classifier import NCDClassifier
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

ds = load_dataset("shunk031/JGLUE", name="MARC-ja")  # type: ignore
train_df = ds["train"].to_pandas()  # type: ignore
val_df = ds["validation"].to_pandas()  # type: ignore

train_df = cast(pd.DataFrame, train_df)
val_df = cast(pd.DataFrame, val_df)

if config.debug:
    # train_df / test_df を label を考慮の上、サンプリングする
    train_df = (
        train_df.groupby("label")
        .apply(lambda x: x.sample(n=1000, random_state=42))
        .reset_index(drop=True)
    )
    val_df = (
        val_df.groupby("label")
        .apply(lambda x: x.sample(n=10, random_state=42))
        .reset_index(drop=True)
    )


print(f"train: {len(train_df)}, test: {len(val_df)}")

print(train_df["label"].value_counts())
print(val_df["label"].value_counts())

X_train_text = train_df["sentence"]
X_test_text = val_df["sentence"]


y_train = train_df["label"].tolist()

X_train = list(map(convert_fn, X_train_text.tolist()))
X_test = list(map(convert_fn, X_test_text.tolist()))

classifier = NCDClassifier(
    n_jobs=-1, k=3, show_progress=True, label_frequency_weighting=True
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# acc の表示

print(accuracy_score(val_df["label"].tolist(), y_pred))

# TP / FP / FN / FP の表示
from sklearn.metrics import confusion_matrix

print(confusion_matrix(val_df["label"].tolist(), y_pred))

# print(classifier._counts)
# print(classifier._scores)
# print(classifier._probabilities)

"""
0.8020870180403255
[[4077  755]
 [ 364  458]]
"""
