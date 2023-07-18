from datasets import load_dataset
from npc_classifier.npc_classifier import NPCClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


def tokenize_str(text: str, tokenizer=tokenizer) -> str:
    encoded_input = ",".join(
        map(str, tokenizer(text, add_special_tokens=False)["input_ids"])  # type: ignore
    )
    return encoded_input


ds = load_dataset("glue", name="sst2")
# ds = load_dataset("shunk031/JGLUE", name="MARC-ja")  # type: ignore
train_df = ds["train"].to_pandas()  # type: ignore
valid_df = ds["validation"].to_pandas()  # type: ignore

# 10000件ランダム抽出
# train_df = train_df.sample(n=10000, random_state=42)  # type: ignore
# valid_df = valid_df.sample(n=1000, random_state=42)  # type: ignore

# label ごとに 1000件ランダム抽出
train_df = train_df.groupby("label").apply(lambda x: x.sample(n=2000, random_state=42)).reset_index(drop=True)  # type: ignore
valid_df = valid_df.groupby("label").apply(lambda x: x.sample(n=100, random_state=42)).reset_index(drop=True)  # type: ignore

print(len(train_df), len(valid_df))

y_train = train_df["label"].tolist()  # type: ignore

X_train = train_df["sentence"].tolist()  # type: ignore
X_test = valid_df["sentence"].tolist()  # type: ignore
# X_train = list(map(tokenize_str, train_df["sentence"].tolist()))  # type: ignore
# X_test = list(map(tokenize_str, valid_df["sentence"].tolist()))  # type: ignore

classifier = NPCClassifier(n_jobs=15, k=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# acc の表示

print(accuracy_score(valid_df["label"].tolist(), y_pred))

# ふつーの
# 0.8516094800141493
# xlm-roberta-large
# 0.853555005305978
