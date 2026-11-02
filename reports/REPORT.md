# Movie Recommendation System - Project Report

## 1. Objective
Build a collaborative-filtering system that recommends movies from user ratings.

## 2. Data
MovieLens-format files: `movies.csv` (movieId, title, genres) and `ratings.csv` (userId, movieId, rating 0.5-5, timestamp). The submitted run uses a synthetic sample (62 movies, 400 users, 7,920 ratings, 68% sparsity) because no internet was available; the pipeline switches to real MovieLens data automatically when it is present.

## 3. Preprocessing
Dropped duplicate and missing rows, kept ratings in [0.5, 5], and filtered users with fewer than 5 ratings and movies with fewer than 3 (iteratively). Extracted release year and genre lists, and converted timestamps to dates.

## 4. Analysis
Ratings cluster around 3-4 stars. Rating counts per movie are long-tailed, so a few titles get most ratings. Genre averages and yearly activity are in `outputs/genre_analysis.csv` and charts 04, 05 and 07.

## 5. Method
Ratings are stored as a sparse user-by-movie matrix and mean-centred per user. Cosine similarity gives user-user or item-item neighbours. A prediction is the user's mean plus the similarity-weighted average of neighbour deviations over the top k = 20 neighbours. Recommendations are the highest predicted ratings among unseen movies.

## 6. Results
| Model |      RMSE     MAE
Global mean    0.8665  0.6886
User-based CF  0.7036  0.5586
Item-based CF  0.6813  0.5446

Both collaborative models beat the baseline; item-based CF was best (about 21% lower RMSE). Example output for User 205 (likes Sci-Fi/Adventure/Comedy): Toy Story, Avatar, Iron Man, Star Wars: A New Hope. These numbers are from synthetic data and will change on real MovieLens data.

## 7. Visualisations
Charts 01-09 in `outputs/charts/`: rating distribution, most-rated and top-rated movies, genre counts and averages, activity distributions, yearly trends, model comparison and per-user recommendations.

## 8. Limitations and future work
Cold-start users and movies are not handled; no content features are used. Next steps: matrix factorisation (SVD/ALS), hybrid content-plus-collaborative models, ranking metrics (Precision@K, NDCG), and a small web UI.
