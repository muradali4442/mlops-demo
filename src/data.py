from typing import Tuple
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_dataset(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    X = iris.data  # columns: sepal length, sepal width, petal length, petal width
    y = iris.target
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
