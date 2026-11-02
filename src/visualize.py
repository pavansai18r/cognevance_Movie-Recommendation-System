"""Charts for EDA and recommendation results."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(movies, ratings, eval_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="rating", data=ratings, ax=ax, color="#4C72B0")
    ax.set(title="Distribution of Ratings", xlabel="Rating", ylabel="Count")
    paths.append(_save(fig, outdir, "01_rating_distribution.png"))

    stats = ratings.groupby("movieId").rating.agg(["count", "mean"]).join(movies.set_index("movieId").title)
    top = stats.sort_values("count", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.title, top["count"], color="#55A868")
    ax.set(title="Top 10 Most-Rated Movies", xlabel="Number of ratings")
    paths.append(_save(fig, outdir, "02_most_rated_movies.png"))

    m = max(10, int(stats["count"].quantile(0.5)))
    best = stats[stats["count"] >= m].sort_values("mean", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(best.title, best["mean"], color="#C44E52")
    ax.set(title=f"Top 10 Highest-Rated Movies (min {m} ratings)", xlabel="Average rating", xlim=(0, 5))
    paths.append(_save(fig, outdir, "03_top_rated_movies.png"))

    g = movies.explode("genre_list")
    g = g[g.genre_list != ""]
    gc = g.genre_list.value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=gc.values, y=gc.index, ax=ax, color="#8172B2")
    ax.set(title="Movies per Genre", xlabel="Number of movies", ylabel="")
    paths.append(_save(fig, outdir, "04_genre_counts.png"))

    rg = ratings.merge(g[["movieId", "genre_list"]], on="movieId")
    ga = rg.groupby("genre_list").rating.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ga.index, ga.values, color="#CCB974")
    ax.set(title="Average Rating by Genre", xlabel="Average rating", xlim=(ga.min() - 0.3, ga.max() + 0.2))
    paths.append(_save(fig, outdir, "05_genre_avg_rating.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ratings.groupby("userId").size().plot.hist(bins=30, ax=axes[0], color="#64B5CD")
    axes[0].set(title="Ratings per User", xlabel="Ratings given", ylabel="Users")
    stats["count"].plot.hist(bins=30, ax=axes[1], color="#4C72B0")
    axes[1].set(title="Ratings per Movie", xlabel="Ratings received", ylabel="Movies")
    paths.append(_save(fig, outdir, "06_activity_distributions.png"))

    ry = ratings.groupby("year").agg(n=("rating", "size"), avg=("rating", "mean"))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(ry.index, ry.n, color="#9bb7d4")
    ax1.set(title="Rating Activity and Average Rating by Year", xlabel="Year", ylabel="Ratings")
    ax2 = ax1.twinx()
    ax2.plot(ry.index, ry.avg, color="#C44E52", marker="o")
    ax2.set(ylabel="Average rating", ylim=(1, 5))
    ax2.grid(False)
    paths.append(_save(fig, outdir, "07_trends_by_year.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    eval_df["RMSE"].plot.bar(ax=ax, color=["#999999", "#4C72B0", "#55A868"], rot=0)
    for i, v in enumerate(eval_df["RMSE"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set(title="Model Comparison (lower RMSE is better)", ylabel="RMSE")
    paths.append(_save(fig, outdir, "08_model_comparison.png"))
    return paths


def recommendation_chart(recs, user_id, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = recs.iloc[::-1]
    ax.barh(r.title, r.predicted_rating, color="#55A868")
    ax.set(title=f"Top Recommendations for User {user_id}", xlabel="Predicted rating", xlim=(0, 5))
    return _save(fig, outdir, f"09_recommendations_user_{user_id}.png")
