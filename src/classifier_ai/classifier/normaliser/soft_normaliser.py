import numpy as np
from typing import Optional, Union
import pandas as pd


class simple_fct_normalizer:
    @staticmethod
    def log_normalization(
        df: pd.DataFrame, columns_to_skip_normalization: list[str] = None
    ) -> pd.DataFrame:
        """
        This function can be utilized to perform log normalization on all columns that do not contain values below 0.
        It also provides the flexibility to exclude specific columns from the normalization process.
        Args:
            df (pd.DataFrame): The dataframe to be processed
            columns_to_skip_normalization (list[str]): list of columns where log normalization should be skipped

        Returns:
            df (pd.DataFrame): The processed dataframe
        """
        df = df.copy()
        # boolean for all columns if in the column a value <= 0 return True else return False
        A = df.columns
        condition_columns = (df[A] > 0).all()
        if columns_to_skip_normalization is None:
            pass
        else:
            condition_columns[columns_to_skip_normalization] = (
                False  # even if the condition is False some columns should not be normalized such as RSI for instance
            )
        df[df.columns[condition_columns]] = np.log(df[df.columns[condition_columns]])
        return df

    @staticmethod
    def Z_score_normalization(
        df: pd.DataFrame,
        mean: Optional[int] = None,
        std: Optional[int] = None,
        row: Optional[int] = None,
        degree: float = 1,
    ) -> np.ndarray:
        """
        Compute the Z-scores for the elements in the input DataFrame, 'matrix', either across rows or columns.

        Parameters:
            df (pd.DataFrame): The input data as a pandas DataFrame.
            mean (Optional[int]): The mean value to use for standardization. If 'row' is provided, it calculates the mean of the specified row. If not provided, the mean is calculated across columns.
            std (Optional[int]): The standard deviation value to use for standardization. If 'row' is provided, it calculates the standard deviation of the specified row. If not provided, the standard deviation is calculated across columns.
            row (Optional[int]): If provided, it specifies a row in the DataFrame for which the Z-scores are computed using the 'mean' and 'std' provided.

        Returns:
            np.ndarray: A NumPy array containing the Z-scores computed based on the specified mean and standard deviation or across rows or columns.

        If 'row' is not provided:
        - If 'mean' and 'std' are not provided, it calculates Z-scores across columns using the mean and standard deviation of the entire DataFrame.
        - If 'mean' and 'std' are provided, it calculates Z-scores using the specified mean and standard deviation values.

        If 'row' is provided:
        - It calculates Z-scores for the specified row using the mean and standard deviation of that row.

        Note: Z-scores are a measure of how many standard deviations an element is from the mean. They are used for standardizing data.
        """
        df = df.copy()

        if row is not None:
            mean, std = np.mean(df.iloc[:, row]), np.std(df.iloc[:, row])
            return (df - mean) / std ** (degree) if std != 0 else (df - mean) / 1

        elif mean is None or 0 >= std or std is None:
            return (df - np.mean(df, axis=0)) / (np.std(df, axis=0) + 1e-6) ** (degree)
        else:
            return (df - mean) / std ** (degree)

    @staticmethod
    def division_relative_normalisation(
        df: pd.DataFrame, row: int = 0, value: bool = True, not_relative: bool = True
    ):
        """
        Normalize selected columns in the input DataFrame by dividing their values by a normalizing column.

        Parameters:
            df (pd.DataFrame): The input data as a pandas DataFrame.
            row (Optional[int]): If provided, it specifies a row in the DataFrame used as the normalizer for the selected columns.
            value (bool): A boolean if The value is True the value used for the normalization would be the first value for the row = row.
            not_relative (bool): A boolean if not_relative is True the ouput is NOT a relative division else it is a normal division

        Returns:
            pd.DataFrame: A pandas DataFrame with selected columns normalized by dividing their values by the values in the 'column_name_normalizer' column.

        This function performs normalization by dividing the values in the selected columns specified in 'list_columns_normalized' by the values in the 'column_name_normalizer' column. The purpose of this operation is to standardize the data or express it not_relative to the normalizing column.
        """
        df = df.copy()
        if value:
            value = df.iloc[0, row] if df.iloc[0, row] != 0 else 1
            df = (
                df.apply(lambda x: x / value)
                if not_relative
                else df.apply(lambda x: x / value) - 1
            )
        else:
            df = (
                df.apply(lambda x: x / df.iloc[:, row])
                if not_relative
                else df.apply(lambda x: x / df.iloc[row]) - 1
            )

        return df

    @staticmethod
    def min_max_normalization(df: pd.DataFrame):
        """
        Normalize selected columns in the input DataFrame by applying min max normalization.

        Parameters:
            df (pd.DataFrame): The input data as a pandas DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame with selected columns normalized by dividing their values by the values in the 'column_name_normalizer' column.

        This function performs normalization by dividing the values in the selected columns specified in 'list_columns_normalized' by the values in the 'column_name_normalizer' column. The purpose of this operation is to standardize the data or express it relative to the normalizing column.
        """
        df = df.copy()
        minimum_df = df.min().min()
        maximum_df = df.max().max()
        df = (
            (df - minimum_df) / (maximum_df - minimum_df)
            if maximum_df - minimum_df != 0
            else (df - minimum_df)
        )
        return df


class complex_fct_normalizer(simple_fct_normalizer):

    def __init__(self):
        super(complex_fct_normalizer, self).__init__()

    def Z_score_log_normalization(
        self,
        df: Union[np.ndarray, pd.DataFrame],
        mean: Optional[int] = None,
        std: Optional[int] = None,
        row: Optional[int] = 0,
        columns_to_skip_normalization: list[str] = None,
        degree: float = 1,
    ):
        """
        Compute Z-scores for elements in the input data after applying logarithmic scaling and optional column-wise normalization.

        Parameters:
            df (Union[np.ndarray, pd.DataFrame]): The input data as a NumPy array or pandas DataFrame.
            mean (Optional[int]): The mean value to use for standardization. If 'row' is provided, it calculates the mean of the specified row. If not provided, the mean is calculated across columns.
            std (Optional[int]): The standard deviation value to use for standardization. If 'row' is provided, it calculates the standard deviation of the specified row. If not provided, the standard deviation is calculated across columns.
            row (Optional[int]): If provided, it specifies a row in the DataFrame for which the Z-scores are computed using the 'mean' and 'std' provided.
            columns_to_skip_normalization (List[str]): A list of column names to skip during normalization.

        Returns:
            np.ndarray: A NumPy array containing the Z-scores computed based on the specified mean and standard deviation or across rows or columns.

        This function first applies logarithmic scaling to the input data and then computes Z-scores:
        1. Logarithmic Scaling: If the input is a pandas DataFrame, the logarithmic scaling is applied to all columns except those specified in 'columns_to_skip_normalization'.
        2. Z-score Calculation: Z-scores are calculated either across rows or columns, depending on the provided 'row' value. If 'row' is not provided:
            - If 'mean' and 'std' are not provided, it calculates Z-scores across columns using the mean and standard deviation of the entire DataFrame.
            - If 'mean' and 'std' are provided, it calculates Z-scores using the specified mean and standard deviation values.
        If 'row' is provided:
        - It calculates Z-scores for the specified row using the mean and standard deviation of that row.
        Note: Z-scores are a measure of how many standard deviations an element is from the mean. They are used for standardizing data.
        """
        df = df.copy()

        return self.Z_score_normalization(
            self.log_normalization(df, columns_to_skip_normalization),
            mean,
            std,
            row,
            degree=degree,
        )

    def Z_score_division_normalization(
        self, df: pd.DataFrame, row: int = 0, degree: float = 1
    ):
        df = df.copy()
        df = self.division_relative_normalisation(
            df, row, value=True, not_relative=True
        )
        return self.Z_score_normalization(df=df, row=row, degree=degree)

    def log_division_normalization(self, df: pd.DataFrame, row: int = 0):
        df = df.copy()
        df = self.division_relative_normalisation(
            df, row, value=True, not_relative=True
        )
        return self.log_normalization(df)
