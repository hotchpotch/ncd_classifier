from typing import Callable, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
import zlib
import numpy as np
from collections import defaultdict
import operator
from joblib import Parallel, delayed
from typing import cast


class ZlibCompressor:
    def __init__(self):
        pass

    def compress_and_get_length(self, text: str) -> int:
        return len(zlib.compress(text.encode("utf-8")))


def combine_texts(text1: str, text2: str) -> str:
    return text1 + text2


def calculate_normalized_distance(len1: int, len2: int, combined_len: int) -> float:
    return (combined_len - min(len1, len2)) / max(len1, len2)


class NPCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        combine_texts: Callable[[str, str], str] = combine_texts,
        calculate_distance: Callable[
            [int, int, int], float
        ] = calculate_normalized_distance,
        k: int = 2,
        n_jobs: int = 1,
    ):
        self.combine_texts = combine_texts
        self.compressor = ZlibCompressor()
        self.calculate_distance = calculate_distance
        self.k = k
        self.n_jobs = n_jobs
        self.train_texts = []
        self.train_labels = []
        self.compressed_train_texts = []

    def fit(self, X: List[str], y: List[int]) -> "NPCClassifier":
        self.train_texts = X
        self.train_labels = y
        compressed_train_texts = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compressor.compress_and_get_length)(x) for x in X
        )
        self.compressed_train_texts = cast(List[int], compressed_train_texts)
        return self

    def predict(self, X: List[str]) -> List[int]:
        predicted_labels = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(x) for x in X
        )
        predicted_labels = cast(List[int], predicted_labels)
        return predicted_labels

    def predict_single(self, text: str) -> int:
        distances = self.calculate_distances_to_train_texts(self.train_texts, text)
        sorted_indices = np.argsort(np.array(distances))
        label_counts = defaultdict(int)
        for j in range(self.k):
            nearest_label = self.train_labels[sorted_indices[j]]
            label_counts[nearest_label] += 1
        sorted_label_counts = sorted(
            label_counts.items(), key=operator.itemgetter(1), reverse=True
        )
        most_frequent_label = sorted_label_counts[0][0]
        return most_frequent_label

    def calculate_distances_to_train_texts(
        self, train_texts: List[str], test_text: str
    ) -> List[float]:
        distances = []
        test_text_compressed_len = self.compressor.compress_and_get_length(test_text)
        for j, train_text in enumerate(train_texts):
            train_text_compressed_len = self.compressed_train_texts[j]
            combined_text_compressed_len = self.compressor.compress_and_get_length(
                self.combine_texts(test_text, train_text)
            )
            distance = self.calculate_distance(
                test_text_compressed_len,
                train_text_compressed_len,
                combined_text_compressed_len,
            )
            distances.append(distance)
        return distances
