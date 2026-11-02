"""Data loading: MovieLens (real) with an offline synthetic fallback.

MovieLens files used: movies.csv (movieId,title,genres) and
ratings.csv (userId,movieId,rating,timestamp).
Put the real files in data/ (see README) and they are used automatically.
"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# (title, year, genres) - used only by the synthetic fallback generator
_SAMPLE_MOVIES = [
    ("The Shawshank Redemption", 1994, "Drama"), ("The Godfather", 1972, "Crime|Drama"),
    ("The Dark Knight", 2008, "Action|Crime|Drama"), ("Pulp Fiction", 1994, "Crime|Drama"),
    ("Forrest Gump", 1994, "Comedy|Drama|Romance"), ("Inception", 2010, "Action|Sci-Fi|Thriller"),
    ("The Matrix", 1999, "Action|Sci-Fi"), ("Fight Club", 1999, "Drama|Thriller"),
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi"), ("Gladiator", 2000, "Action|Adventure|Drama"),
    ("Toy Story", 1995, "Animation|Children|Comedy"), ("The Lion King", 1994, "Animation|Children|Drama"),
    ("Finding Nemo", 2003, "Animation|Children|Comedy"), ("Shrek", 2001, "Animation|Comedy|Fantasy"),
    ("Up", 2009, "Animation|Adventure|Children"), ("Inside Out", 2015, "Animation|Children|Comedy"),
    ("Titanic", 1997, "Drama|Romance"), ("The Notebook", 2004, "Drama|Romance"),
    ("La La Land", 2016, "Comedy|Drama|Musical|Romance"), ("Pride & Prejudice", 2005, "Drama|Romance"),
    ("The Hangover", 2009, "Comedy"), ("Superbad", 2007, "Comedy"),
    ("Bridesmaids", 2011, "Comedy|Romance"), ("Step Brothers", 2008, "Comedy"),
    ("Jurassic Park", 1993, "Action|Adventure|Sci-Fi|Thriller"), ("Avatar", 2009, "Action|Adventure|Sci-Fi"),
    ("Star Wars: A New Hope", 1977, "Action|Adventure|Sci-Fi"), ("Blade Runner 2049", 2017, "Drama|Sci-Fi"),
    ("The Avengers", 2012, "Action|Adventure|Sci-Fi"), ("Iron Man", 2008, "Action|Adventure|Sci-Fi"),
    ("The Silence of the Lambs", 1991, "Crime|Horror|Thriller"), ("Se7en", 1995, "Crime|Mystery|Thriller"),
    ("Get Out", 2017, "Horror|Mystery|Thriller"), ("The Conjuring", 2013, "Horror|Mystery|Thriller"),
    ("A Quiet Place", 2018, "Drama|Horror|Sci-Fi"), ("Psycho", 1960, "Horror|Thriller"),
    ("Lord of the Rings: Fellowship", 2001, "Adventure|Fantasy"), ("Harry Potter: Sorcerer's Stone", 2001, "Adventure|Children|Fantasy"),
    ("Pirates of the Caribbean", 2003, "Action|Adventure|Fantasy"), ("The Princess Bride", 1987, "Adventure|Comedy|Fantasy|Romance"),
    ("Schindler's List", 1993, "Drama|War"), ("Saving Private Ryan", 1998, "Action|Drama|War"),
    ("Braveheart", 1995, "Action|Drama|War"), ("1917", 2019, "Action|Drama|War"),
    ("Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi"), ("John Wick", 2014, "Action|Crime|Thriller"),
    ("Die Hard", 1988, "Action|Thriller"), ("Casablanca", 1942, "Drama|Romance|War"),
    ("Parasite", 2019, "Comedy|Drama|Thriller"), ("Spirited Away", 2001, "Animation|Adventure|Fantasy"),
    ("Coco", 2017, "Animation|Children|Musical"), ("Whiplash", 2014, "Drama|Musical"),
    ("The Social Network", 2010, "Drama"), ("Knives Out", 2019, "Comedy|Crime|Mystery"),
    ("Back to the Future", 1985, "Adventure|Comedy|Sci-Fi"), ("Ratatouille", 2007, "Animation|Children|Comedy"),
    ("Mamma Mia!", 2008, "Comedy|Musical|Romance"), ("Zodiac", 2007, "Crime|Mystery|Thriller"),
    ("The Prestige", 2006, "Drama|Mystery|Sci-Fi"), ("Gone Girl", 2014, "Drama|Mystery|Thriller"),
    ("Dune", 2021, "Action|Adventure|Sci-Fi"), ("Oppenheimer", 2023, "Drama|History"),
]


def make_sample_data(n_users=400, seed=42):
    """Create a MovieLens-shaped dataset with genre-driven taste (offline fallback)."""
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame(
        [(i + 1, f"{t} ({y})", g) for i, (t, y, g) in enumerate(_SAMPLE_MOVIES)],
        columns=["movieId", "title", "genres"],
    )
    genres = sorted({g for gs in movies.genres for g in gs.split("|")})
    quality = rng.normal(0, 0.5, len(movies))            # latent movie quality
    popularity = rng.pareto(1.2, len(movies)) + 0.3      # long-tailed popularity
    popularity /= popularity.sum()
    rows = []
    for u in range(1, n_users + 1):
        taste = {g: rng.normal(0, 0.9) for g in genres}   # per-user genre taste
        bias = rng.normal(0, 0.3)
        n = int(np.clip(rng.lognormal(2.9, 0.5), 8, len(movies)))
        picks = rng.choice(len(movies), n, replace=False, p=popularity)
        for m in picks:
            gs = movies.genres[m].split("|")
            score = 3.4 + bias + quality[m] + np.mean([taste[g] for g in gs]) + rng.normal(0, 0.4)
            r = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))
            ts = int(rng.integers(946684800, 1700000000))
            rows.append((u, int(movies.movieId[m]), r, ts))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return movies, ratings


def download_movielens(data_dir):
    """Download MovieLens 'latest-small' (100k ratings) into data_dir."""
    with urllib.request.urlopen(ML_URL, timeout=30) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in ("movies.csv", "ratings.csv"):
        with z.open(f"ml-latest-small/{name}") as src, open(os.path.join(data_dir, name), "wb") as dst:
            dst.write(src.read())


def load_data(data_dir="data", try_download=True):
    """Return (movies, ratings, source_label)."""
    os.makedirs(data_dir, exist_ok=True)
    mp, rp = (os.path.join(data_dir, f) for f in ("movies.csv", "ratings.csv"))
    if not (os.path.exists(mp) and os.path.exists(rp)) and try_download:
        try:
            download_movielens(data_dir)
        except Exception as exc:  # offline or blocked network
            print(f"[data] MovieLens download failed ({exc.__class__.__name__}); using sample data.")
    if os.path.exists(mp) and os.path.exists(rp):
        movies, ratings = pd.read_csv(mp), pd.read_csv(rp)
        label = "sample" if ratings.userId.nunique() == 400 and len(movies) == len(_SAMPLE_MOVIES) else "MovieLens"
        return movies, ratings, label
    movies, ratings = make_sample_data()
    movies.to_csv(mp, index=False)
    ratings.to_csv(rp, index=False)
    return movies, ratings, "sample"
