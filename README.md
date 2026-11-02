# Movie Recommendation System (Cognevance - Level 1)

A collaborative-filtering recommender built with Python. It cleans a MovieLens-style ratings dataset, analyses ratings and genres, trains user-based and item-based k-NN models, evaluates them on held-out ratings, and prints top-N recommendations per user.

## Project workflow
1. **Collect**: `src/data_loader.py` loads MovieLens `ml-latest-small` (auto-download) or any `movies.csv` + `ratings.csv` placed in `data/`.
2. **Clean**: `src/preprocess.py` removes duplicates, missing and out-of-range ratings, and sparse users/movies; extracts release year and genre lists.
3. **Analyse**: rating distribution, genre statistics, popularity and yearly trends (`outputs/*.csv`).
4. **Model**: `src/recommender.py` implements mean-centred cosine-similarity k-NN (user-based and item-based).
5. **Evaluate**: 80/20 split, RMSE and MAE vs a global-mean baseline.
6. **Recommend**: top-N unseen movies for different users, similar-movie lookup, popularity baseline.
7. **Visualise**: 9 charts in `outputs/charts/`.

## Technologies
Python 3, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn.

## Run
```bash
pip install -r requirements.txt
python main.py                          # random 3 users
python main.py --users 1 50 100 --top-n 10 --k 20
```

## Dataset note
`data/movies.csv` and `data/ratings.csv` currently hold a **synthetic MovieLens-format sample** (62 movies, 400 users, ~7.9k ratings), because the environment used to build this had no internet access. To use the real data, delete those two files and run `python main.py` on a connected machine (it downloads MovieLens `ml-latest-small`), or copy the files from https://grouplens.org/datasets/movielens/latest/ into `data/`. Then re-run and commit the new outputs/screenshots.

## Structure
```
main.py            pipeline entry point
src/               data_loader, preprocess, recommender, visualize
data/              movies.csv, ratings.csv, ratings_clean.csv
models/            item_cf_model.pkl
outputs/           recommendations.txt/.csv, evaluation, analysis CSVs, charts/
reports/           REPORT.md
screenshots/       add terminal/chart screenshots here
```

## results 
                  RMSE     MAE
Global mean     0.8665    0.6886
User-based CF   0.7036    0.5586
Item-based CF   0.6813    0.5446

See `outputs/recommendations.txt` for per-user recommendations and `outputs/charts/` for visualisations.
