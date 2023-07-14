from typing import Callable, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
import zlib
import numpy as np
from collections import defaultdict
import operator
from joblib import Parallel, delayed


class DefaultCompressor:
    def __init__(self):
        pass

    def get_compressed_len(self, text: str) -> int:
        return len(zlib.compress(text.encode("utf-8")))


def agg_func(text1: str, text2: str) -> str:
    return text1 + text2


def dis_func(len1: int, len2: int, len12: int) -> float:
    return (len12 - min(len1, len2)) / max(len1, len2)


class NPCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        agg_func: Callable[[str, str], str] = agg_func,
        dis_func: Callable[[int, int, int], float] = dis_func,
        k: int = 2,
        n_jobs: int = 1,
    ):
        self.agg_func = agg_func
        self.compressor = DefaultCompressor()
        self.dis_func = dis_func
        self.k = k
        self.n_jobs = n_jobs
        self.train_data = None
        self.train_label = None
        self.train_compressed = None

    def fit(self, X: List[str], y: List[int]) -> "NPCClassifier":
        self.train_data = X
        self.train_label = y
        self.train_compressed = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compressor.get_compressed_len)(x) for x in X
        )
        return self

    def predict(self, X: List[str]) -> List[int]:
        y_pred = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(x) for x in X
        )
        return y_pred

    def predict_single(self, x: str) -> int:
        distance4i = self.calc_dis_single_multi(self.train_data, x)
        sorted_idx = np.argsort(np.array(distance4i))
        pred_labels = defaultdict(int)
        for j in range(self.k):
            pred_l = self.train_label[sorted_idx[j]]
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(
            pred_labels.items(), key=operator.itemgetter(1), reverse=True
        )
        most_label = sorted_pred_lab[0][0]
        return most_label

    def calc_dis_single_multi(self, train_data: List[str], datum: str) -> List[float]:
        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)
        for j, t2 in enumerate(train_data):
            t2_compressed = self.train_compressed[j]
            t1t2_compressed = self.compressor.get_compressed_len(
                self.agg_func(datum, t2)
            )
            distance = self.dis_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)
        return distance4i
