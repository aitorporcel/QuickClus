import pandas as pd
import numpy as np

def check_is_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Requires a pandas DataFrame as input")


