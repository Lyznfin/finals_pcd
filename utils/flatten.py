import pandas as pd

def flatten_columns(features_df: pd.DataFrame):
    flattened = pd.DataFrame()
    for col in features_df.columns:
        flattened = pd.concat([flattened, pd.DataFrame(features_df[col].tolist(), columns=[f"{col}_{i*45}" for i in range(len(features_df[col][0]))])], axis=1)
    return flattened