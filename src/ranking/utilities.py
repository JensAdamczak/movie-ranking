import numpy as np
import pandas as pd
import tabulate as t

from ranking import ranking_methods as rank

def wide_to_long(df):

    df_long = pd.melt(
        df, 
        id_vars=['timestamp', 'choice'], 
        value_vars=['tconst_1', 'tconst_2'],
        value_name='tconst'
    )

    conds = df_long['variable'].str[-1].astype(int) == df_long['choice']
    df_long = (
        df_long
        .assign(won = np.where(conds, 1, 0))
        .loc[:, ['tconst', 'won']]
    )

    return df_long 


def ascii_hist(x, scale=20, symbol="+"):

    counts, bins = np.histogram(x)
    bin_centers = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
    counts_scale = np.round(counts / max(counts) * scale, 0)

    max_digits_counts = max([len(str(c)) for c in counts])
    max_digits_bins = max([len(str(round(b, 1))) for b in bin_centers])
    for i in reversed(range(len(bin_centers))):
        str_count = f"{counts[i]:<{max_digits_counts}}"
        str_bin = f"{round(bin_centers[i], 1):<{max_digits_bins}}"
        str_symbol = f"{symbol*int(counts_scale[i])}"

        print(f"{str_count} | {str_bin} | {str_symbol}")


def ascii_plot(x, y, scale=10):
    # 2D ascii scatter plot
    #char_map = list("""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'.""")

    points = list(zip(x, y))
    df_points = pd.DataFrame(points, columns = ['x', 'y'])

    bins_x, cats_x = pd.cut(df_points['x'], scale, labels=list(range(1, scale+1)), retbins=True)
    x_mid = cats_x[:-1] + (cats_x[1:, ] - cats_x[:-1]) / 2
    bins_y, cats_y = pd.cut(df_points['y'], scale, labels=list(range(1, scale+1)), retbins=True)
    y_mid = cats_y[:-1] + (cats_y[1:, ] - cats_y[:-1]) / 2
    df_points['bin_x'] = bins_x
    df_points['bin_y'] = bins_y 

    max_digits_x = max([len(str(round(x, 2))) for x in x_mid])
    max_digits_y = max([len(str(round(y, 2))) for y in y_mid])

    df_gr = (
        df_points
        .groupby(['bin_x', 'bin_y'])
        .agg(n=('bin_x', 'size'))
        .reset_index()
    )
    df_gr = df_gr[df_gr['n'] != 0]

    for yit in reversed(range(1, scale+1)):
        str_x = f"{round(x_mid[yit-1], 2):<{max_digits_x}}"
        str_y = f"{round(y_mid[yit-1], 2):<{max_digits_y}}"

        idx = df_gr.loc[df_gr['bin_y'] == yit, 'bin_x'].tolist()
        str_symbol = "".join(['$' if e in idx else '.' for e in range(1, scale+1)])

        print(f"{str_x} | {str_y} | {str_symbol}")


def print_movie_stats(df_wins, df_movies):
    
    df_wins_long = wide_to_long(df_wins)

    df_agg = (
        df_wins_long
        .groupby('tconst')
        .agg(n_competitions=('won', 'size'),
            n_wins=('won', 'sum'))
        .reset_index()
        .assign(win_ratio = lambda df: round(df['n_wins'] / df['n_competitions'], 3))
        .merge(df_movies, on='tconst')
    )

    print("")
    print("> Distribution of nr of competitions")
    print("")
    ascii_hist(df_agg['n_competitions'], scale=50)

    print("")
    print("> Distribution of win ratios")
    print("")
    ascii_hist(df_agg['win_ratio'], scale=50)

    df_most_voted = (
        df_agg
        .sort_values(by=['n_competitions', 'n_wins'], ascending=[False, False])
        .loc[:, ['title', 'n_competitions', 'n_wins']]
        .head(10)
    )
    print("")
    print("> Most voted movies")
    print("")
    print(t.tabulate(df_most_voted, headers='keys', tablefmt='simple', showindex=False))
    df_most_wins = (
        df_agg
        .sort_values(by=['n_wins', 'n_competitions'], ascending=[False, False])
        .loc[:, ['title', 'n_wins', 'n_competitions']]
        .head(10)
    )
    print("")
    print("> Movies with most wins")
    print("")
    print(t.tabulate(df_most_wins, headers='keys', tablefmt='simple', showindex=False))

    df_best_ratio = (
        df_agg
        .sort_values(by=['win_ratio', 'n_competitions'], ascending=[False, False])
        .loc[:, ['title', 'win_ratio', 'n_competitions']]
        .head(10)
    )
    print("")
    print("> Movies with best win ratio")
    print("")
    print(t.tabulate(df_best_ratio, headers='keys', tablefmt='simple', showindex=False))


def reg_check(df_wins):
    np.random.seed(1201210)
    n_comp = df_wins.shape[0]
    idx_train = np.random.rand(n_comp) < 0.8
    df_comp_train = df_wins[idx_train]
    df_comp_test = df_wins[~idx_train]
    print(f"Training share: {round(len(df_comp_train) / len(df_wins), 3)}")
    print(f"Test share:     {round(len(df_comp_test) / len(df_wins), 3)}")

    reg_list = np.arange(0, 50, 0.05) + 0.01
    rmse_list = []
    for reg in reg_list:
        rmse_list.append(rank.regcheck_bradley_terry(df_comp_train, df_comp_test, reg))
    df_reg = pd.DataFrame({'regularization':reg_list, 'rmse':rmse_list})

    df_plot = df_reg.sort_values(by="rmse", ascending=True).head(100)
    ascii_plot(df_plot['regularization'], df_plot['rmse'], scale=100)
