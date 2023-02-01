import pandas as pd
import numpy as np

class DataTransformation():
    """
    Class to transform the data frame.
    """
    def transform(self, dataframe_path: str) -> None:
        """
        Makes DF into new format.
        :param dataframe_path: string of the df
        """
        # read in df
        df = pd.read_excel(dataframe_path)

        # melt the hours into a column
        df = df.melt(id_vars=["PRICES"],
                     var_name="Hour",
                     value_name="Price (Euros)")

        # change col names
        df = df.rename(columns={"PRICES": "Date"})

        # sort df by date
        df = df.sort_values(by=["Date", "Hour"])

        # reset index
        df = df.reset_index(drop=True)

        old_hours = df.Hour.unique().tolist()
        new_hours = [*range(1, 25, 1)]
        df['Hour'] = df['Hour'].replace(old_hours, new_hours)

        # Make new cols/features based on data

        # time related
        df["Day"] = pd.DatetimeIndex(df["Date"]).day
        df["Month"] = pd.DatetimeIndex(df["Date"]).month
        df["Year"] = pd.DatetimeIndex(df["Date"]).year
        df["BusinessDay"] = df.Date.apply(self._is_business_day)
        df["Season"] = df["Month"].apply(lambda x: "Winter" if x in [9, 10, 11, 12, 1] else "Summer")
        df["Night"] = df["Hour"].apply(lambda x: True if x in [24, 1, 2, 3, 4, 5, 6, 7] else False)

        # Price trend related
        df["Trend_abs"] = df["Price (Euros)"].diff()
        down_treshold = float(df["Trend_abs"].where(df["Trend_abs"] < 0).quantile([0.5]))
        up_treshold = float(df["Trend_abs"].where(df["Trend_abs"] > 0).quantile([0.5]))

        df["Trend"] = df["Trend_abs"]
        df["Trend"] = df["Trend"].mask(df["Trend_abs"] > 0, 3)
        df["Trend"] = df["Trend"].mask(df["Trend_abs"] == 0.0, 2)
        df["Trend"] = df["Trend"].mask(df["Trend_abs"] < 0, 1)

        df["Trend"] = df["Trend"].mask(df["Trend_abs"] >= up_treshold, 4)
        df["Trend"] = df["Trend"].mask(df["Trend_abs"] <= down_treshold, 0)
        df[["Trend_abs", "Trend"]]

        df["Trend"] = df["Trend"].apply(self._sign)
        df["BusinessDay"] = df["BusinessDay"].apply(self._bool_to_num)
        df["Night"] = df["Night"].apply(self._bool_to_num)
        df["Season"] = df["Season"].apply(self._season_to_num)

        df.to_csv(dataframe_path.replace(".xlsx", "") + "_transformed.csv")


    def _is_business_day(self, date):
        return bool(len(pd.bdate_range(date, date)))

    # if slope negative: 0; if slope 0; 1, if slope positive: 2
    def _sign(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        else:
            return 2

    # if True: 1; if False: 0
    def _bool_to_num(self, x):
        if x:
            return 1
        else:
            return 0

    # if Summer: 1; if Winter: 0
    def _season_to_num(self, x):
        if x == "Summer":
            return 1
        elif x == "Winter":
            return 0