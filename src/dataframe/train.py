import pandas as pd
from dataframe.base import Preprocessor


class TrainPreprocessor(Preprocessor):
    def __init__(self, df):
        self.df = df
        self.processed = self.preprocess()

    def preprocess(self) -> pd.DataFrame:
        res = self.df.copy()
        res["clean_title"] = res["clean_title"].fillna("NaN")
        res["is.clean_title"] = res["clean_title"] == "Yes"
        res["age"] = (2024 - res["model_year"]).map(lambda x: max(x, 1))
        res["milage_per_year"] = res["milage"] / res["age"]
        res["had_accident"] = (
            res["accident"] == "At least 1 accident or damage reported"
        )
        res["avg_price"] = res.groupby(["brand", "model"])["price"].transform("mean")
        res["model_year"] = res["model_year"]
        return res

    def X(self):
        return self.df[
            [
                "is.clean_title",
                "milage_per_year",
                "had_accident",
                "avg_price",
                "model_year",
            ]
        ]

    def y(self):
        return self.df["price"]
