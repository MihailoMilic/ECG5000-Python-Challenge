import numpy as np
import polars as pl

from data_loader import load_training_data


def main():
    data = load_training_data()

    df = pl.DataFrame(
        data,
        schema=["label"] + [f"sample_{i}" for i in range(1, data.shape[1])],
        orient="row",
    ).with_columns(pl.col("label").cast(pl.Int64))

    print(df.head())

if __name__ == "__main__":
    main()