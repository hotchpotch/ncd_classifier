# NPC Classifier

NPC Classifier is a Python library that implements the method proposed in the paper ["Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors"](https://aclanthology.org/2023.findings-acl.426/). This method is a non-parametric alternative to deep neural networks for text classification, using a combination of a simple compressor like gzip with a k-nearest-neighbor classifier. It is easy to use, lightweight, and does not require any training parameters, making it suitable for low-resource languages and few-shot settings.

This code was implemented with reference to [https://github.com/bazingagin/npc_gzip](https://github.com/bazingagin/npc_gzip).

This library is designed with a scikit-learn interface, making it familiar and straightforward for users with experience in scikit-learn.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

This library can be installed using pip:

```bash
pip install npc_classifier
```

## Usage

Here is a simple example of how to use the NPC Classifier:

```python
from npc_classifier import NPCClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Assume X_train, y_train, and X_test are your preprocessed training and test data

classifier = NPCClassifier(n_jobs=-1, k=3, show_progress=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## License

This code is licensed under the MIT License.

# Citation

```
@inproceedings{jiang-etal-2023-low,
    title = "{``}Low-Resource{''} Text Classification: A Parameter-Free Classification Method with Compressors",
    author = "Jiang, Zhiying  and
      Yang, Matthew  and
      Tsirlin, Mikhail  and
      Tang, Raphael  and
      Dai, Yiqin  and
      Lin, Jimmy",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.426",
    pages = "6810--6828",
    abstract = "Deep neural networks (DNNs) are often used for text classification due to their high accuracy. However, DNNs can be computationally intensive, requiring millions of parameters and large amounts of labeled data, which can make them expensive to use, to optimize, and to transfer to out-of-distribution (OOD) cases in practice. In this paper, we propose a non-parametric alternative to DNNs that{'}s easy, lightweight, and universal in text classification: a combination of a simple compressor like \textit{gzip} with a $k$-nearest-neighbor classifier. Without any training parameters, our method achieves results that are competitive with non-pretrained deep learning methods on six in-distribution datasets.It even outperforms BERT on all five OOD datasets, including four low-resource languages. Our method also excels in the few-shot setting, where labeled data are too scarce to train DNNs effectively.",
}
```
