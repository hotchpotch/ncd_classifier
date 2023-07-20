from __future__ import annotations
from typing import Callable
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import defaultdict
import operator
from concurrent.futures import ProcessPoolExecutor
from .compressors import COMPRESSORS
from tqdm import tqdm
from typing import Callable, Sequence
from numpy import ndarray
import numpy as np
import psutil


def default_concatenate_fn(
    data1: str | Sequence[int], data2: str | Sequence[int]
) -> str | Sequence[int]:
    if isinstance(data1, str) and isinstance(data2, str):
        return data1 + " " + data2
    elif isinstance(data1, ndarray) and isinstance(data2, ndarray):
        return np.concatenate((data1, data2), axis=0)  # type: ignore
    elif isinstance(data1, Sequence) and isinstance(data2, Sequence):
        return list(data1) + list(data2)  # type: ignore
    else:
        raise ValueError("data1 and data2 must be the same type.")


def compute_normalized_distance(len1: int, len2: int, combined_len: int) -> float:
    """
    Calculates the normalized compression distance between two pieces of data.
    """
    return (combined_len - min(len1, len2)) / max(len1, len2)


def _softmax(x: Sequence[float]) -> Sequence[float]:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class NPCClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier based on the Normalized Compression Distance (NCD).
    """

    def __init__(
        self,
        concatenate_fn: Callable[
            [str | Sequence[int], str | Sequence[int]], str | Sequence[int]
        ] = default_concatenate_fn,
        compute_distance: Callable[
            [int, int, int], float
        ] = compute_normalized_distance,
        compress_len_fn: Callable[[str | Sequence[int]], int] = COMPRESSORS["zlib"],
        k: int = 3,
        n_jobs: int = -1,
        show_progress: bool = False,
    ):
        """
        Initializes the NPCClassifier.
        """
        self.concatenate_fn = concatenate_fn
        self.compute_distance = compute_distance
        self.compress_len_fn = compress_len_fn
        self.k = k
        if n_jobs == -1:
            num_physical_cores = psutil.cpu_count(logical=False)
            self.n_jobs = max(1, num_physical_cores - 1)
        else:
            self.n_jobs = max(1, n_jobs)
        self.show_progress = show_progress
        self.train_data = []
        self.train_labels = []
        self.compressed_train_data = []
        self._scores = []
        self._counts = []
        self._probabilities = []

    def fit(
        self, X: Sequence[str] | Sequence[Sequence[int]], y: Sequence[int]
    ) -> NPCClassifier:
        """
        Fits the model using the training data.
        """
        self.train_data = X
        self.train_labels = y

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.compress_len_fn, x) for x in X]

            if self.show_progress:
                futures = tqdm(futures, total=len(futures), desc="Fitting")

            self.compressed_train_data = [future.result() for future in futures]

        return self

    def predict(self, X: Sequence[str] | Sequence[Sequence[int]]) -> Sequence[int]:
        """
        Predicts the labels of the given data.
        """
        self._scores = []
        self._counts = []
        self._probabilities = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.predict_single, x) for x in X]

            if self.show_progress:
                futures = tqdm(futures, total=len(futures), desc="Predicting")

            predicted_labels = [future.result() for future in futures]

        return predicted_labels

    def predict_single(self, data: str | Sequence[int]) -> int:
        """
        Predicts the label of a single instance.
        """
        distances = self.calculate_distances_to_train_data(self.train_data, data)
        sorted_indices = np.argsort(np.array(distances))
        label_counts = defaultdict(float)
        for label in self.train_labels:
            label_counts[label] += 1
        total_label_counts = sum(label_counts.values())
        for label in label_counts:
            label_counts[label] /= total_label_counts
            label_counts[label] = 1 / label_counts[label]
        nearest_label_counts = defaultdict(int)
        nearest_label_scores = defaultdict(float)
        for j in range(self.k):
            nearest_label = self.train_labels[sorted_indices[j]]
            nearest_label_counts[nearest_label] += 1
        for label in label_counts:
            # Taking into account the occurrence rate
            nearest_label_scores[label] = (
                nearest_label_counts[label] * label_counts[label]
            )

        sorted_label_counts = sorted(
            nearest_label_scores.items(), key=operator.itemgetter(1), reverse=True
        )
        most_frequent_label = sorted_label_counts[0][0]
        softmax_probabilities = list(
            _softmax([count for _, count in sorted_label_counts])
        )

        self._scores.append(dict(nearest_label_scores))
        self._counts.append(dict(nearest_label_counts))
        self._probabilities.append(softmax_probabilities)

        return most_frequent_label

    def calculate_distances_to_train_data(
        self,
        train_data: Sequence[str] | Sequence[Sequence[int]],
        test_data: str | Sequence[int],
    ) -> Sequence[float]:
        """
        Calculates the distances from a test instance to all training instances.
        """
        distances = []
        test_data_compressed_len = self.compress_len_fn(test_data)
        for j, data in enumerate(train_data):
            train_data_compressed_len = self.compressed_train_data[j]
            combined_data_compressed_len = self.compress_len_fn(
                self.concatenate_fn(test_data, data)
            )
            distance = self.compute_distance(
                test_data_compressed_len,
                train_data_compressed_len,
                combined_data_compressed_len,
            )
            distances.append(distance)
        return distances
