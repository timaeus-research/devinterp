import csv
import os

import numpy as np
import pandas as pd
import pytest

from devinterp.ops.logging import (
    CompositeLogger,
    CsvLogger,
    DataFrameLogger,
    StdLogger,
    WandbLogger,
)


def test_csv_logger():
    csv_path = "test.csv"
    csv_logger = CsvLogger(csv_path, ["metric1", "metric2"])

    csv_logger.log({"metric1": 1, "metric2": 2}, step=0, commit=True)

    try:
        with open(csv_path, "r") as file:
            reader = csv.reader(file)
            assert list(reader) == [["0", "1", "2"]]
    finally:
        os.remove(csv_path)


def test_dataframe_logger():
    df_logger = DataFrameLogger([0, 10, 20], ["metric1", "metric2"])
    df_logger.log({"metric1": 1, "metric2": 2}, step=0)

    assert df_logger.df.loc[0].equals(
        pd.Series({"step": 0, "metric1": np.nan, "metric2": np.nan})
    )

    df_logger.log({"metric1": 3}, step=20, commit=False)
    df_logger.log({"metric2": 4}, step=20, commit=True)

    print(df_logger.df)

    assert df_logger.df.loc[0].equals(
        pd.Series({"step": 0, "metric1": 1.0, "metric2": 2.0})
    )
    assert df_logger.df.loc[2].equals(
        pd.Series({"step": 20, "metric1": 3.0, "metric2": 4.0})
    )


def test_wandb_logger(mocker):
    mock_log = mocker.patch("wandb.log")
    wandb_logger = WandbLogger("proj", "entity")
    wandb_logger.log({"metric": 1}, step=0)
    mock_log.assert_called_with({"metric": 1}, step=0)


def test_std_logger(mocker):
    mock_logging = mocker.patch("logging.Logger.info")
    std_logger = StdLogger("proj")
    std_logger.log({"metric": 1}, step=0)
    mock_logging.assert_called()


def test_composite_logger(mocker):
    mock_logger1 = mocker.MagicMock()
    mock_logger2 = mocker.MagicMock()
    composite_logger = CompositeLogger([mock_logger1, mock_logger2])

    composite_logger.log({"metric": 1}, step=0)

    mock_logger1.log.assert_called_with({"metric": 1}, step=0)
    mock_logger2.log.assert_called_with({"metric": 1}, step=0)
