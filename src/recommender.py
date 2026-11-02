"""Collaborative filtering: user-based and item-based k-NN with cosine similarity."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CFRecommender:
    def __init__(self, ratings, movies, mode="item", k=20):
        assert mode in ("user", "item")
        self.mode, self.k, self.movies = mode, k, movies.set_index("movieId")
        self.users = np.sort(ratings.userId.unique())
        self.items = np.sort(ratings.movieId.unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {m: i for i, m in enumerate(self.items)}
        r = ratings.userId.map(self.u_idx).values
        c = ratings.movieId.map(self.i_idx).values
        self.R = csr_matrix((ratings.rating.values, (r, c)), shape=(len(self.users), len(self.items)))
        self.global_mean = ratings.rating.mean()
        counts = np.maximum(self.R.getnnz(axis=1), 1)
        self.user_mean = np.asarray(self.R.sum(axis=1)).ravel() / counts
        # mean-centre each user's ratings so similarity reflects taste, not generosity
        C = self.R.copy().astype(float)
        for u in range(C.shape[0]):
            s, e = C.indptr[u], C.indptr[u + 1]
            C.data[s:e] -= self.user_mean[u]
        self.C = C
        base = C if mode == "user" else C.T.tocsr()
        S = cosine_similarity(base, dense_output=True)
        np.fill_diagonal(S, 0)
        self.S = S

    # --- prediction ---------------------------------------------------------
    def _predict_row(self, u):
        """Predicted ratings for every item for user index u."""
        if self.mode == "user":
            sim = self.S[u]
            nbrs = np.argsort(sim)[-self.k:]
            w = sim[nbrs]
            Cn = self.C[nbrs].toarray()
            rated = (Cn != 0) | (self.R[nbrs].toarray() > 0)
            num = (w[:, None] * Cn).sum(0)
            den = (np.abs(w)[:, None] * rated).sum(0) + 1e-9
            return self.user_mean[u] + num / den
        rated_idx = self.R[u].indices
        rated_val = self.C[u].toarray().ravel()
        preds = np.full(len(self.items), self.user_mean[u])
        sub = self.S[:, rated_idx]                      # items x rated items
        if self.k < sub.shape[1]:                       # keep top-k neighbours per item
            thresh = np.sort(sub, axis=1)[:, -self.k][:, None]
            sub = np.where(sub >= thresh, sub, 0)
        num = sub @ rated_val[rated_idx]
        den = np.abs(sub).sum(1) + 1e-9
        return self.user_mean[u] + num / den

    def predict(self, user_id, movie_id):
        if user_id not in self.u_idx or movie_id not in self.i_idx:
            return self.global_mean
        p = self._predict_row(self.u_idx[user_id])[self.i_idx[movie_id]]
        return float(np.clip(p, 0.5, 5.0))

    def recommend(self, user_id, n=10):
        """Top-n unseen movies for a user, with predicted rating."""
        u = self.u_idx[user_id]
        p = np.clip(self._predict_row(u), 0.5, 5.0)
        seen = set(self.R[u].indices)
        order = [i for i in np.argsort(-p) if i not in seen][:n]
        out = pd.DataFrame({"movieId": self.items[order], "predicted_rating": p[order].round(2)})
        out = out.join(self.movies[["title", "genres"]], on="movieId")
        return out[["title", "genres", "predicted_rating"]].reset_index(drop=True)

    def similar_movies(self, title_substr, n=5):
        """Item-item neighbours (works in item mode)."""
        m = self.movies[self.movies.title.str.contains(title_substr, case=False, regex=False)]
        if m.empty:
            return pd.DataFrame()
        i = self.i_idx[m.index[0]]
        Sm = cosine_similarity(self.C.T.tocsr())[i]
        Sm[i] = -1
        top = np.argsort(-Sm)[:n]
        out = self.movies.loc[self.items[top], ["title", "genres"]].copy()
        out["similarity"] = Sm[top].round(3)
        return out.reset_index(drop=True)


def popularity_baseline(train, movies, n=10, m=10):
    """Bayesian-average popularity list (non-personalised baseline)."""
    stats = train.groupby("movieId").rating.agg(["mean", "count"])
    C = train.rating.mean()
    stats["score"] = (stats["count"] * stats["mean"] + m * C) / (stats["count"] + m)
    return stats.sort_values("score", ascending=False).join(movies.set_index("movieId")[["title", "genres"]]).head(n)


def evaluate(train, test, movies, k=20, max_test=3000):
    """RMSE / MAE of user-CF, item-CF and a global-mean baseline on held-out ratings."""
    test = test.sample(min(len(test), max_test), random_state=0)
    results = {}
    gm = train.rating.mean()
    results["Global mean"] = _metrics(test.rating.values, np.full(len(test), gm))
    for mode in ("user", "item"):
        model = CFRecommender(train, movies, mode=mode, k=k)
        preds = []
        for uid, grp in test.groupby("userId"):
            if uid not in model.u_idx:
                preds += [(i, gm) for i in grp.index]
                continue
            row = model._predict_row(model.u_idx[uid])
            for idx, mid in zip(grp.index, grp.movieId):
                p = row[model.i_idx[mid]] if mid in model.i_idx else gm
                preds.append((idx, float(np.clip(p, 0.5, 5.0))))
        p = pd.Series(dict(preds)).loc[test.index].values
        results[f"{mode.title()}-based CF"] = _metrics(test.rating.values, p)
    return pd.DataFrame(results, index=["RMSE", "MAE"]).T.round(4)


def _metrics(y, p):
    return [float(np.sqrt(np.mean((y - p) ** 2))), float(np.mean(np.abs(y - p)))]
