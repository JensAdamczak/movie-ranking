import click
import numpy as np
import pandas as pd

# import sys
# sys.path.append('/Users/adamczak/GitHub/movie-ranking-1/src/')
from voting.ComparisonGui import ComparisonGui
from pathlib import Path

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')

def select_movie_pairs(df_movies, df_comp, n=10):
    """Select n movie comparisons."""

    # We need fresh comparisons that we haven't voted for.
    df_movies_comp = (
        df_comp
        .loc[(df_comp['timestamp'] == 0) & (df_comp['choice'] == 0), :]
    )

    # Pick a random one.
    df_movies_comp = (
        df_movies_comp
        .iloc[np.random.randint(0, df_movies_comp.shape[0], n), :]
        .reset_index(drop=True)
    )

    # We need the right format for the comparison UI.
    df_movies_comp = (
        df_movies_comp
        .merge((
            df_movies
            .assign(tconst_1=lambda x: x['tconst'])
            .assign(title_1=lambda x: x['title'])
            .drop(['tconst', 'title'], axis=1)
        ), on='tconst_1')
        .merge((
            df_movies
            .assign(tconst_2=lambda x: x['tconst'])
            .assign(title_2=lambda x: x['title'])
            .drop(['tconst', 'title'], axis=1)
        ), on='tconst_2')
        .loc[:, ['tconst_1', 'title_1', 'tconst_2', 'title_2']]
    )

    return df_movies_comp


@click.command()
@click.option('--data-path', default=None, 
              help='Directory for movie data and comparison files.')
@click.option('--comparison-file', default='comparison_results.csv', 
              help='CSV file with movie comparison results.')
@click.option('--movie-file', default='watched_movies.csv',
              help="CSV file with movie metadata.")
@click.option('--n', default=10,
              help='Number of movie comparisons to be made.')
def main(data_path, comparison_file, movie_file, n):

    # Calculate path to files.
    data_directory = Path(data_path) if data_path else default_data_directory
    csv_comp = data_directory.joinpath(comparison_file)
    csv_movies = data_directory.joinpath(movie_file)

    df_comp = pd.read_csv(csv_comp, sep=";")
    df_movies = pd.read_csv(csv_movies, sep=";")
    #df_comp = pd.read_csv("/Users/adamczak/GitHub/movie-ranking-1/data/comparison_results.csv", #sep=';')
    #df_movies = pd.read_csv("/Users/adamczak/GitHub/movie-ranking-1/data/watched_movies.csv", #sep=";")

    df_movies_comp = select_movie_pairs(df_movies, df_comp, n)
    try:
        comp_gui = ComparisonGui(df_movies_comp)

        # Before updating comparison_results.csv check if ordering is correct (smaller tconst first before) in comparison.csv and df_results
        # Should we write a backup in votes/ dir? 
        if len(np.where(comp_gui.df_results['tconst_1'] > comp_gui.df_results['tconst_2'])) > 0:

            idx = ['tconst_1', 'tconst_2']

            df_comp = df_comp.set_index(idx)
            df_comp.update(comp_gui.df_results.set_index(idx))
            df_comp = df_comp.reset_index()[['timestamp', 'tconst_1', 'tconst_2', 'choice']]
            df_comp.to_csv(csv_comp, index=False, float_format='%.0f', sep=";")
            print(f"Movie comparison completed and {csv_comp} updated.")
        else:
            print("Movies not in correct order in comparison output data frame.")
    except AttributeError:
        print("Comparison could not be completed.")

    print("Done.")

if __name__ == '__main__':
    main()
