"""Week 2: Advanced Data Visualization and Storytelling with Python
Dataset: public Iris flower dataset (150 observations; measurements in centimetres).
Run: pip install pandas numpy matplotlib scikit-learn
     python analysis.py
This script loads the Iris dataset from scikit-learn, creates five visualizations,
and writes chart PNG files and CSV summaries into the current directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

OUT = "week2_outputs"
os.makedirs(OUT, exist_ok=True)
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"sepal length (cm)":"sepal_length", "sepal width (cm)":"sepal_width", "petal length (cm)":"petal_length", "petal width (cm)":"petal_width", "target":"species_id"})
df["species"] = df["species_id"].map(dict(enumerate(iris.target_names)) )
df = df.drop(columns=["species_id"]).drop_duplicates().dropna()
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df.to_csv(os.path.join(OUT, "iris_cleaned.csv"), index=False)
df[features].corr().to_csv(os.path.join(OUT, "correlation_matrix.csv"))

# 1: scatter plot
plt.figure(figsize=(8,5))
for name, group in df.groupby("species"):
    plt.scatter(group.petal_length, group.petal_width, label=name, alpha=.75)
plt.xlabel("Petal length (cm)"); plt.ylabel("Petal width (cm)")
plt.title("Petal dimensions by Iris species"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_petal_scatter.png"), dpi=160); plt.close()

# 2: box plot
plt.figure(figsize=(8,5))
plt.boxplot([df.loc[df.species==s, "petal_length"] for s in sorted(df.species.unique())], tick_labels=sorted(df.species.unique()))
plt.ylabel("Petal length (cm)"); plt.title("Petal length distribution by species")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "02_petal_boxplot.png"), dpi=160); plt.close()

# 3: grouped mean bar chart
means = df.groupby("species")[features].mean()
means.plot(kind="bar", figsize=(9,5)); plt.ylabel("Mean measurement (cm)")
plt.title("Average measurements by species"); plt.xticks(rotation=0)
plt.legend(title="Measurement", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "03_species_means.png"), dpi=160); plt.close("all")

# 4: correlation heatmap
corr = df[features].corr()
plt.figure(figsize=(7,6)); im=plt.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
plt.xticks(range(4), ["Sepal L","Sepal W","Petal L","Petal W"], rotation=30, ha="right")
plt.yticks(range(4), ["Sepal L","Sepal W","Petal L","Petal W"])
for i in range(4):
    for j in range(4): plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
plt.colorbar(im, label="Pearson correlation"); plt.title("Correlation between measurements")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "04_correlation_heatmap.png"), dpi=160); plt.close()

# 5: overlapping histograms
plt.figure(figsize=(8,5))
for name, group in df.groupby("species"):
    plt.hist(group.petal_length, bins=12, alpha=.45, label=name)
plt.xlabel("Petal length (cm)"); plt.ylabel("Number of flowers")
plt.title("Petal length frequency by species"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_petal_histogram.png"), dpi=160); plt.close()
df.describe(include="all").to_csv(os.path.join(OUT, "descriptive_statistics.csv"))
print("Done. Charts and CSV outputs saved in", OUT)
