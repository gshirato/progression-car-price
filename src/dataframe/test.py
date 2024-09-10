import pandas as pd
from dataframe.base import Preprocessor


class TestPreprocessor(Preprocessor):
    def __init__(self, df, train_preprocessed):
        self.df = df
        self.train_preprocessed = train_preprocessed
        self.avg_price_from_brand_model = self.get_avg_price_from_brand_model(
            self.train_preprocessed
        )
        self.avg_price_from_brand = self.get_avg_price_from_brand(
            self.train_preprocessed
        )

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
        res["model_year"] = res["model_year"]
        res["avg_price"] = self.add_avg_price(res)

        return res

    def add_avg_price(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(
            lambda x: self.retrieve_value(
                x["brand"],
                x["model"],
                self.avg_price_from_brand_model,
                self.avg_price_from_brand,
            ),
            axis=1,
        )

    def retrieve_value(
        self, brand: str, model: str, avg_price_from_brand_model: dict, from_brand: dict
    ) -> float:
        """
        Retrieve a value from `avg_price` dictionary using `brand` and `model` as keys.
        If `model` is not found in the dictionary, use `brand` to retrieve the value.
        If `brand` key is not found, return None.
        """
        if (brand, model) in avg_price_from_brand_model:
            return avg_price_from_brand_model[(brand, model)]
        if brand in from_brand:
            return from_brand[brand]

        return None

    def get_avg_price_from_brand_model(self, df: pd.DataFrame) -> dict:
        return df.groupby(["brand", "model"])["avg_price"].mean().to_dict()

    def get_avg_price_from_brand(self, df: pd.DataFrame) -> dict:
        return df.groupby("brand")["avg_price"].mean().to_dict()

    def X(self):
        return self.processed[
            [
                "is.clean_title",
                "milage_per_year",
                "had_accident",
                "avg_price",
                "model_year",
            ]
        ]
