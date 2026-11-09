import numpy as np
import polars as pl

def main():
    data = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")

    df = pl.DataFrame(
        data,
        schema=["label"] + [f"sample_{i}" for i in range(1, data.shape[1])],
        orient="row",
    ).with_columns(pl.col("label").cast(pl.Int64))

    print(df.head())

if __name__ == "__main__":
    main()