import pandas as pd
import numpy as np
from .util.utils import check_nan,convert_to_dataframe

def rga(y: list, yhat: list):
        """
        RANK GRADUATION ACCURACY (RGA) MEASURE 
        Function for the RGA measure computation.

        Parameters
        ----------
        y : list
                A list of actual values.
        yhat : list
                A list of predicted values.

        Returns
        -------
        float
                The RGA value.
        """
            
        # Convert inputs to DataFrames and concatenate them
        y, yhat = convert_to_dataframe(y, yhat)
        # check the length
        if y.shape != yhat.shape:
                raise ValueError("y and yhat should have the same shape.")
        df = pd.concat([y, yhat], axis=1)
        df.columns = ["y", "yhat"]
        # check for missing values
        check_nan(y, yhat)
              
        # Rank yhat values
        df['ryhat'] = df['yhat'].rank(method="min")

        # Group by ryhat and calculate mean of y (support)
        support = df.groupby('ryhat')['y'].mean().reset_index(name='support')

        # Merge support back to the original dataframe
        df = pd.merge(df, support, on="ryhat", how="left")

        # Create the rord column by directly assigning 'support' where ryhat matches
        df['rord'] = df['support']
        
        # Sort df by yhat to get correct ordering for ystar
        df = df.sort_values(by="yhat").reset_index(drop=True)

        # Get ystar in the same order as rord in the sorted dataframe
        ystar = df['rord'].values

        # Create the index array I
        I = np.arange(len(df))

        # Calculate conc, dec (descending order of y) and inc (ascending order of y)
        conc = np.sum(I * ystar)
        sorted_y = np.sort(df['y'])  # y sorted in ascending order
        dec = np.sum(I * sorted_y[::-1])  # y sorted in descending order
        inc = np.sum(I * sorted_y)

        # Compute the RGA
        RGA = (conc - dec) / (inc - dec)

        return RGA