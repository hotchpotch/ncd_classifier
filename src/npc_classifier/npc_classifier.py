from typing import Callable, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
import zlib
import numpy as np
from collections import defaultdict
import operator
from joblib import Parallel, delayed
from typing import cast


def zlib_compression_length(text: str) -> int:
    """
    Uses zlib to compress a string and returns the length of the compressed string.
    Args:
        text (str): Text to be compressed.

    Returns:
        int: Length of the compressed text.
    """
    return len(zlib.compress(text.encode("utf-8")))


def concatenate_texts(text1: str, text2: str) -> str:
    """
    Combines two texts with a space in between.
    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        str: Combined text.
    """
    return text1 + " " + text2


def compute_normalized_distance(len1: int, len2: int, combined_len: int) -> float:
    """
    Calculates the normalized compression distance between two strings.
    Args:
        len1 (int): Length of the first compressed text.
        len2 (int): Length of the second compressed text.
        combined_len (int): Length of the combined compressed text.

    Returns:
        float: Normalized compression distance.
    """
    return (combined_len - min(len1, len2)) / max(len1, len2)


class NPCClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier based on the Normalized Compression Distance (NCD).
    """

    def __init__(
        self,
        concatenate_texts: Callable[[str, str], str] = concatenate_texts,
        compute_distance: Callable[
            [int, int, int], float
        ] = compute_normalized_distance,
        compress_len_fn: Callable[[str], int] = zlib_compression_length,
        k: int = 2,
        n_jobs: int = 1,
    ):
        """
        Initializes the NPCClassifier.
        Args:
            concatenate_texts (Callable): Function to concatenate two texts.
            compute_distance (Callable): Function to compute the distance.
            compress_len_fn (Callable): Function to compute the length of compressed text.
            k (int): Number of neighbors for k-NN.
            n_jobs (int): Number of jobs to run in parallel.
        """
        self.concatenate_texts = concatenate_texts
        self.compute_distance = compute_distance
        self.compress_len_fn = compress_len_fn
        self.k = k
        self.n_jobs = n_jobs
        self.train_texts = []
        self.train_labels = []
        self.compressed_train_texts = []

    def fit(self, X: List[str], y: List[int]) -> "NPCClassifier":
        """
        Fits the model using the training data.
        Args:
            X (List[str]): Training data.
            y (List[int]): Labels of the training data.

        Returns:
            NPCClassifier: The fitted model.
        """
        self.train_texts = X
        self.train_labels = y
        compressed_train_texts = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compress_len_fn)(x) for x in X
        )
        self.compressed_train_texts = cast(List[int], compressed_train_texts)
        return self

    def predict(self, X: List[str]) -> List[int]:
        """
        Predicts the labels of the given data.
        Args:
            X (List[str]): Data to predict.

        Returns:
            List[int]: Predicted labels.
        """
        predicted_labels = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(x) for x in X
        )
        predicted_labels = cast(List[int], predicted_labels)
        return predicted_labels

    def predict_single(self, text: str) -> int:
        """
        Predicts the label of a single instance.
        Args:
            text (str): Single instance to predict.

        Returns:
            int: Predicted label.
        """
        distances = self.calculate_distances_to_train_texts(self.train_texts, text)
        sorted_indices = np.argsort(np.array(distances))
        label_counts = defaultdict(int)
        for label in self.train_labels:
            label_counts[label] += 1
        total_label_counts = sum(label_counts.values())
        for label in label_counts:
            label_counts[label] /= total_label_counts
            # 逆数にする
            label_counts[label] = 1 / label_counts[label]
        # print(label_counts)
        nearest_label_counts = defaultdict(int)
        for j in range(self.k):
            nearest_label = self.train_labels[sorted_indices[j]]
            nearest_label_counts[nearest_label] += 1
            # print(most_frequent_label)
        # label_counts を掛ける
        for label in label_counts:
            nearest_label_counts[label] *= label_counts[label]
        # print(nearest_label_counts)

        sorted_label_counts = sorted(
            nearest_label_counts.items(), key=operator.itemgetter(1), reverse=True
        )
        most_frequent_label = sorted_label_counts[0][0]
        return most_frequent_label

    def calculate_distances_to_train_texts(
        self, train_texts: List[str], test_text: str
    ) -> List[float]:
        """
        Calculates the distances from a test instance to all training instances.
        Args:
            train_texts (List[str]): Training data.
            test_text (str): Single test instance.

        Returns:
            List[float]: List of distances.
        """
        distances = []
        test_text_compressed_len = self.compress_len_fn(test_text)
        for j, train_text in enumerate(train_texts):
            train_text_compressed_len = self.compressed_train_texts[j]
            combined_text_compressed_len = self.compress_len_fn(
                self.concatenate_texts(test_text, train_text)
            )
            distance = self.compute_distance(
                test_text_compressed_len,
                train_text_compressed_len,
                combined_text_compressed_len,
            )
            distances.append(distance)
        return distances
