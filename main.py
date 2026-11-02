"""Movie Recommendation System - end-to-end pipeline.

Usage:  python main.py [--data-dir data] [--users 1 7 42] [--top-n 10] [--k 20]
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import clean
from src.recommender import CFRecommender, evaluate, popularity_baseline
from src.visualize import make_all, recommendation_chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--users", type=int, nargs="*", default=None)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--k", type=int, default=20, help="neighbours")
    args = ap.parse_args()
    out = "outputs"
    os.makedirs(f"{out}/charts", exist_ok=True)

    # 1-2. collect + clean
    movies, ratings, source = load_data(args.data_dir)
    movies, ratings, log = clean(movies, ratings)
    print(f"Dataset: {source} | {log}")
    pd.DataFrame([log]).to_csv(f"{out}/cleaning_summary.csv", index=False)
    ratings.drop(columns=["date"]).to_csv(f"{args.data_dir}/ratings_clean.csv", index=False)

    # 3. analysis
    stats = ratings.groupby("movieId").rating.agg(["count", "mean"]).join(movies.set_index("movieId").title)
    stats.sort_values("count", ascending=False).head(20).to_csv(f"{out}/top_movies_by_popularity.csv")
    g = movies.explode("genre_list").merge(ratings, on="movieId")
    g.groupby("genre_list").rating.agg(["count", "mean"]).round(3).sort_values("mean", ascending=False) \
        .to_csv(f"{out}/genre_analysis.csv")
    print("\nSparsity: %.2f%%" % (100 * (1 - len(ratings) / (ratings.userId.nunique() * ratings.movieId.nunique()))))

    # 4. evaluation (hold out 20% of ratings)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    test = test[test.userId.isin(train.userId) & test.movieId.isin(train.movieId)]
    ev = evaluate(train, test, movies, k=args.k)
    ev.to_csv(f"{out}/model_evaluation.csv")
    print("\nModel evaluation (held-out ratings):\n", ev)

    # 5. recommendations from the item-based model trained on all data
    model = CFRecommender(ratings, movies, mode="item", k=args.k)
    os.makedirs('models', exist_ok=True)
    pickle.dump(model, open('models/item_cf_model.pkl', 'wb'))
    users = args.users or list(np.random.default_rng(1).choice(model.users, 3, replace=False))
    lines = [f"Movie Recommendations (item-based CF, k={args.k}, source={source})\n" + "=" * 70]
    all_recs = []
    for i, u in enumerate(users):
        u = int(u)
        liked = ratings[ratings.userId == u].sort_values("rating", ascending=False).head(5) \
            .merge(movies, on="movieId")[["title", "genres", "rating"]]
        recs = model.recommend(u, args.top_n)
        lines += [f"\nUser {u} - favourite rated movies:", liked.to_string(index=False),
                  f"\nUser {u} - top {args.top_n} recommendations:", recs.to_string(index=False)]
        all_recs.append(recs.assign(userId=u))
        if i == 0:
            recommendation_chart(recs, u, f"{out}/charts")
    anchor = stats.sort_values("count", ascending=False).title.iloc[0]
    sim = model.similar_movies(anchor, 5)
    lines += [f"\nMovies similar to '{anchor}':", sim.to_string(index=False)]
    pop = popularity_baseline(ratings, movies, args.top_n)
    lines += ["\nNon-personalised popularity baseline:", pop[["title", "genres", "score"]].round(2).to_string()]
    text = "\n".join(lines)
    open(f"{out}/recommendations.txt", "w").write(text)
    pd.concat(all_recs).to_csv(f"{out}/recommendations.csv", index=False)
    print("\n" + text)

    # 6. charts
    paths = make_all(movies, ratings, ev, f"{out}/charts")
    print("\nSaved charts:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
