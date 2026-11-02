"""Cleaning and feature preparation."""
import pandas as pd


def clean(movies, ratings, min_user_ratings=5, min_movie_ratings=3):
    """Remove duplicates/missing/out-of-range values and sparse users/movies."""
    log = {"ratings_raw": len(ratings), "movies_raw": len(movies)}
    movies = movies.drop_duplicates("movieId").dropna(subset=["title"]).copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings[ratings.rating.between(0.5, 5.0)]
    ratings = ratings[ratings.movieId.isin(movies.movieId)]

    # iterate: filtering users can push movies below threshold and vice versa
    while True:
        n = len(ratings)
        ratings = ratings[ratings.groupby("userId").movieId.transform("count") >= min_user_ratings]
        ratings = ratings[ratings.groupby("movieId").userId.transform("count") >= min_movie_ratings]
        if len(ratings) == n:
            break

    movies = movies[movies.movieId.isin(ratings.movieId)].copy()
    movies["year"] = movies.title.str.extract(r"\((\d{4})\)\s*$").astype(float)
    movies["genre_list"] = movies.genres.replace("(no genres listed)", "").str.split("|")
    ratings = ratings.copy()
    ratings["date"] = pd.to_datetime(ratings.timestamp, unit="s")
    ratings["year"] = ratings.date.dt.year
    log.update(ratings_clean=len(ratings), movies_clean=len(movies),
               users=ratings.userId.nunique())
    return movies.reset_index(drop=True), ratings.reset_index(drop=True), log
