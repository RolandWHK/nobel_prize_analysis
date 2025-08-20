# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path="data/nobel.csv") -> pd.DataFrame:
    """Load Nobel Prize dataset."""
    return pd.read_csv(path)


def most_common_gender_country(df: pd.DataFrame):
    """Return most common gender and birth country among Nobel laureates."""
    top_gender = df["sex"].value_counts().idxmax()
    top_country = df["birth_country"].value_counts().idxmax()
    return top_gender, top_country


def us_winners_by_decade(df: pd.DataFrame):
    """Compute ratio of US-born Nobel laureates per decade and plot trend."""
    df = df.copy()
    df["usa_winners"] = df["birth_country"] == "United States of America"
    df["decade"] = (10 * np.floor(df["year"] / 10)).astype(int)
    summary = df.groupby("decade", as_index=False)["usa_winners"].mean()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary, x="decade", y="usa_winners", marker="o")
    plt.title("Ratio of US-born Nobel Prize Winners by Decade")
    plt.xlabel("Decade")
    plt.ylabel("US-born Winner Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "usa_winners_ratio.png")
    plt.close()

    max_decade = summary.loc[summary["usa_winners"].idxmax(), "decade"]
    return int(max_decade)


def highest_female_ratio(df: pd.DataFrame):
    """Return decade and category with highest female laureate ratio."""
    df = df.copy()
    df["decade"] = (10 * np.floor(df["year"] / 10)).astype(int)
    df["female_laureates"] = df["sex"] == "Female"

    grouped = df.groupby(["category", "decade"], as_index=False)["female_laureates"].mean()
    top = grouped.loc[grouped["female_laureates"].idxmax()]
    return {int(top["decade"]): top["category"]}


def first_female_winner(df: pd.DataFrame):
    """Return the first female Nobel laureate and her category."""
    first = df[df["sex"] == "Female"].sort_values("year").iloc[0]
    return first["full_name"], first["category"], int(first["year"])


def multiple_winners(df: pd.DataFrame):
    """Return list of laureates who won more than once."""
    counts = df["full_name"].value_counts()
    return list(counts[counts >= 2].index)


if __name__ == "__main__":
    nobel_df = load_data()

    top_gender, top_country = most_common_gender_country(nobel_df)
    print("Most common gender:", top_gender)
    print("Most common birth country:", top_country)

    max_decade_usa = us_winners_by_decade(nobel_df)
    print("Decade with highest US-born ratio:", max_decade_usa)

    female_peak = highest_female_ratio(nobel_df)
    print("Highest female laureate ratio:", female_peak)

    name, category, year = first_female_winner(nobel_df)
    print(f"First female laureate: {name} ({category}, {year})")

    repeat_laureates = multiple_winners(nobel_df)
    print("Laureates with multiple prizes:", repeat_laureates)
