import os
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt


def generate_eda_report(df: pd.DataFrame, output_dir: str = "reports/eda"):
    """
    Generate basic EDA reports: info, describe, missing values, dtypes, correlation heatmap.
    Saves outputs under the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    # DataFrame info
    buf = io.StringIO()
    df.info(buf=buf)
    with open(os.path.join(output_dir, "info.txt"), "w", encoding='utf-8') as f:
        f.write(buf.getvalue())
    # describe all columns
    df.describe(include="all").to_csv(os.path.join(output_dir, "describe.csv"))
    # missing values
    df.isnull().sum().to_csv(os.path.join(output_dir, "missing_values.csv"))
    # dtypes
    pd.DataFrame(df.dtypes, columns=["dtype"]).to_csv(os.path.join(output_dir, "dtypes.csv"))
    # correlation heatmap (numeric only)
    corr = df.select_dtypes(include=["number"]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close() 