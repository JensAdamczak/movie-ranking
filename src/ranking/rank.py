import click
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from ranking import ranking_methods as rank
from ranking import utilities as u

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')

'''
df_test = pd.DataFrame(
    data = {
        'timestamp': [1, 2, 3, 4],
        'tconst_1': ['tt1', 'tt2', 'tt2', 'tt3'],
        'tconst_2': ['tt2', 'tt4', 'tt3', 'tt5'],
        'choice': [1, 1, 2, 2]
    }
)
'''

@click.command()
@click.option('--data-path', default=None, 
              help='Directory for data files: comparison results and movie files.')
@click.option('--comparison-file', default='comparison_results.csv', 
              help='CSV file with movie comparison results.')
@click.option('--movie-file', default='watched_movies.csv',
              help="CSV file with movie metadata.")
@click.option('--method', default="rank_bradley_terry",
help= """Method name must be one of the following: \
rank_baseline, rank_directly, rank_bradley_terry, rank_svc, reg_check.""")
def main(data_path, comparison_file, movie_file, method):

    # Calculate path to files.
    data_directory = Path(data_path) if data_path else default_data_directory
    csv_comp = data_directory.joinpath(comparison_file)
    csv_movies = data_directory.joinpath(movie_file)

    df_comp = pd.read_csv(csv_comp, sep=";")
    df_wins = df_comp[df_comp['choice'].isin([1, 2])].copy()
    df_movies = pd.read_csv(csv_movies, sep=";")

    ranking_method = {
        'rank_baseline': (rank.rank_baseline, [df_wins]), 
        'rank_directly': (rank.rank_directly, [df_wins, 2]),
        'rank_bradley_terry': (rank.rank_bradley_terry, [df_wins, df_comp, 12]),
        'rank_svc': (rank.rank_SVC, [df_wins,   12]),
        'reg_check': (u.reg_check, [df_wins])
    }

    ranking_function, ranking_paras = ranking_method[method]
    df_ranking_res = ranking_function(*ranking_paras)

    try:
        df_top_rank = (
            df_ranking_res
            .sort_values(by=['score'], ascending=False)
            .merge(df_movies, on='tconst')
            .loc[:, ['title', 'score']]
            .head(10)
        )

        u.print_movie_stats(df_wins, df_movies)

        print("")
        print("> Top ranked movies")
        print("")
        print(tabulate(df_top_rank, headers='keys', tablefmt='simple', showindex=False))
    except AttributeError:
        print(f"No ranking for method {method}.")

if __name__ == '__main__':
    main()
