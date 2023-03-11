import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm
from ranking import utilities as u


def rank_baseline(df):
    df_long = u.wide_to_long(df)
    df_wins = (
        df_long
        .groupby('tconst')
        .agg(n=('won', 'size'),
             wins=('won', 'sum'))
        .reset_index()
    )
    df_wins['score'] = df_wins['wins'] / df_wins['n']
    
    return df_wins[['tconst', 'score']]


def rank_directly(df, p=2):

    df_comp = df.copy()
    # Count number of comparisons that each movie has
    df_long = u.wide_to_long(df_comp)
    df_n = (
        df_long
        .groupby('tconst')
        .agg(n=('won', 'size'))
        .reset_index()
    )

    # Reorder the columns in case movie 2 won
    tconst_1_ = np.where(df_comp['choice'] == 1, df_comp['tconst_1'], df_comp['tconst_2'])
    tconst_2_ = np.where(df_comp['choice'] == 1, df_comp['tconst_2'], df_comp['tconst_1'])
    df_comp['tconst_1'] = tconst_1_
    df_comp['tconst_2'] = tconst_2_
    df_comp['choice'] = 1

    df_comp = df_comp.merge(df_n, how="inner", left_on='tconst_1', right_on='tconst')

    # Replace movie identifiers with index
    unique_movies = df_long['tconst'].unique()
    d_movie_idx = {m:i for (i, m) in enumerate(unique_movies)}

    df_comp['tconst_1_idx'] = df_comp['tconst_1'].map(d_movie_idx)
    df_comp['tconst_2_idx'] = df_comp['tconst_2'].map(d_movie_idx)

    # Fill preference matrix
    n_movies = len(unique_movies)
    r0 = np.ones(n_movies)

    row = df_comp['tconst_1_idx']
    col = df_comp['tconst_2_idx']
    data = df_comp['choice'] / df_comp['n']
    A = sparse.coo_matrix(
        (data, (row, col)), 
        shape=(n_movies, n_movies)
    )
    M = np.ones((n_movies, n_movies)) * 1E-6
    A = A+M

    # Calculate ranking
    # A**2 * r0 gives the average win percentage of teams that were defeated 
    # (include all games in the denominator)
    #v_res = (A*A) * r0

    v = np.linalg.matrix_power(A, p).dot(r0)
    v_norm = np.linalg.norm(v)
    v_res = np.array(v / v_norm).flatten()

    df_out = pd.DataFrame({
        'tconst': unique_movies,
        'score': v_res
    })

    return df_out


def score_bradley_terry(df, reg):

    df_comp = df.copy()

    df_long = u.wide_to_long(df_comp)
    unique_movies = df_long['tconst'].unique()
    d_movie_idx = {m:i for (i, m) in enumerate(unique_movies)}

    cols_1 = list(df_comp['tconst_1'].map(d_movie_idx))
    rows_1 = [i for i in range(len(cols_1))]
    data_1 = list(np.ones(len(cols_1)))
    cols_2 = list(df_comp['tconst_2'].map(d_movie_idx))
    rows_2 = [i for i in range(len(cols_2))]
    data_2 = list(np.ones(len(cols_2)) * -1)
    rows = rows_1 + rows_2
    cols = cols_1 + cols_2
    data = data_1 + data_2

    X = sparse.coo_matrix(
        (data, (rows, cols)), 
        shape=(len(rows_1), len(unique_movies))
    )
    # Remove first movie to avoid the dummy trap
    X = X.toarray()[:, 1:len(unique_movies)]
    y = np.where(df_comp['choice'] == 2, 0, df_comp['choice']) 

    #mod_logreg = LogisticRegression(penalty='none', fit_intercept=False, max_iter=1000)
    mod_logreg = LogisticRegression(fit_intercept=False, max_iter=1000, C=reg)
    #print(mod_logreg)
    mod_logreg.fit(X, y)

    rank_score = np.exp(np.insert(mod_logreg.coef_[0], 0, 0))
    #rank_score = np.insert(mod_logreg.coef_[0], 0, 0)
    df_out = pd.DataFrame({
        'tconst': unique_movies,
        'score': rank_score
    })

    return df_out


def rank_bradley_terry(df, df_competitions, reg=15):

    df_res = score_bradley_terry(df, reg)

    score_dict = {
        movie:score for movie, score in zip(df_res['tconst'], df_res['score'])
    }
    
    df_1 = (
        df_competitions
        .loc[:, ['tconst_1', 'tconst_2']]
        .rename(columns={'tconst_1':'tconst', 'tconst_2':'tconst_opp'})
    )
    df_2 = (
        df_competitions
        .loc[:, ['tconst_2', 'tconst_1']]
        .rename(columns={'tconst_2':'tconst', 'tconst_1':'tconst_opp'})
    )
    df_scores = pd.concat([df_1, df_2], ignore_index=True)

    df_scores['score'] = df_scores['tconst'].map(score_dict)
    df_scores['score_opp'] = df_scores['tconst_opp'].map(score_dict)
    df_scores['ewp'] = df_scores['score'] / (df_scores['score'] + df_scores['score_opp'])

    df_ranks = (
        df_scores
        .groupby('tconst')
        .agg(
            score=('ewp', 'mean'),
            # exp_wins=('ewp', 'sum'),
        )
        .reset_index()
        .sort_values(by='score', ascending=False)
    )
    df_ranks['rank'] = np.arange(1, df_ranks.shape[0] + 1)

    return df_ranks[['tconst', 'score']]


def regcheck_bradley_terry(df_train, df_test, reg, plot=False):
    df_res = score_bradley_terry(df_train, reg)

    df_matches = (
        df_test
        .merge(
            df_res
            .assign(tconst_1=lambda df: df['tconst'])
            .assign(score_1=lambda df:df['score'])
            .loc[:, ['tconst_1', 'score_1']], 
            how="inner", 
            on="tconst_1"
        )
        .merge(
            df_res
            .assign(tconst_2=lambda df: df['tconst'])
            .assign(score_2=lambda df:df['score'])
            .loc[:, ['tconst_2', 'score_2']], 
            how="inner", 
            on="tconst_2"
        )
        .assign(ewp_1=lambda df: df['score_1'] / (df['score_1'] + df['score_2']))
        .assign(ewp_2=lambda df: df['score_2'] / (df['score_1'] + df['score_2']))
    )

    df_wins = (
        df_matches
        .assign(won_1=np.where(df_matches['choice'] == 1, 1, 0))
        .assign(won_2=np.where(df_matches['choice'] == 2, 1, 0))
    )
    df_1 = (
        df_wins
        .loc[:, ['tconst_1', 'won_1', 'score_1', 'ewp_1']]
        .rename(columns={'tconst_1':'tconst', 'won_1':'won', 'score_1':'score', 'ewp_1':'ewp'})
    )
    df_2 = (
        df_wins
        .loc[:, ['tconst_2', 'won_2', 'score_2', 'ewp_2']]
        .rename(columns={'tconst_2':'tconst', 'won_2':'won', 'score_2':'score', 'ewp_2':'ewp'})
    )
    df_comb = pd.concat([df_1, df_2], ignore_index=True)

    df_win_stats = (
        df_comb
        .groupby('tconst')
        .agg(
            expected_wins=('ewp', 'sum'),
            actual_wins=('won', 'sum'),
            competitions=('won', 'size')
        )
        .reset_index(drop=True)
    )

    rmse = mean_squared_error(df_win_stats['actual_wins'], 
                              df_win_stats['expected_wins'], 
                              squared=True)

    if plot:
        print("")
        u.ascii_hist(df_win_stats['expected_wins'])
        print("")
        u.ascii_hist(df_win_stats['actual_wins'])
        print("")
        u.ascii_plot(df_win_stats['expected_wins'], df_win_stats['actual_wins'], 20)

    return rmse


def rank_SVC(df, reg=15):

    df_comp = df.copy()

    df_long = u.wide_to_long(df_comp)
    unique_movies = df_long['tconst'].unique()
    d_movie_idx = {m:i for (i, m) in enumerate(unique_movies)}

    cols_1 = list(df_comp['tconst_1'].map(d_movie_idx))
    rows_1 = [i for i in range(len(cols_1))]
    data_1 = list(np.ones(len(cols_1)))
    cols_2 = list(df_comp['tconst_2'].map(d_movie_idx))
    rows_2 = [i for i in range(len(cols_2))]
    data_2 = list(np.ones(len(cols_2)) * -1)
    rows = rows_1 + rows_2
    cols = cols_1 + cols_2
    data = data_1 + data_2

    X = sparse.coo_matrix(
        (data, (rows, cols)), 
        shape=(len(rows_1), len(unique_movies))
    )
    # Remove first movie to avoid the dummy trap
    X = X.toarray()[:, 1:len(unique_movies)]
    y = df_comp['choice'].map(lambda x: -1 if x == 2 else 1)

    #mod_svm = LinearSVC(fit_intercept=True, loss="hinge", max_iter=10000, dual=False)
    mod_svm = svm.SVC(kernel="linear", C=reg)
    print(mod_svm)
    mod_svm.fit(X, y)

    rank_score = np.insert(mod_svm.coef_[0], 0, 0) + mod_svm.intercept_[0]
    df_out = pd.DataFrame({
        'tconst': unique_movies,
        'score': rank_score
    })

    return df_out
