import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Preprocessor(ABC):
    df: pd.DataFrame

    @abstractmethod
    def preprocess(self):
        pass