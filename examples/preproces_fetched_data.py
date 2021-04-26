"""
Fetch data related to coins and store them into data folder
"""
from coinpy.util import get_all_folders, get_all_data
import argparse
import os


def fetch_files(args):
    # 1. Get all first level folders under path.
    for i in get_all_folders(args.path):
        # 2. For each fetch data folder, collect csv files.
        for j in get_all_folders(i):
            if args.suffix in j:
                coin_csv_file = j[j.rfind('/') + 1:-4]
                yield coin_csv_file


def preprocessing(args):
    try:
        os.mkdir(args.path_to_store)
    except FileExistsError:
        """ File exists do not need to create a folder """
    for i in fetch_files(args):
        df = get_all_data(data=i, path=args.path)
        # Remove suffix at saving
        key = args.suffix.replace('.csv', '')
        i = i.replace(key, '')
        df.to_csv(f'{args.path_to_store}/' + i + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='../FetchedData',
                        help='Path of folder containing fetch data. A fetch data corresponds to a folder containing coin related info in csv format')
    parser.add_argument("--path_to_store", default='../ProcessedData')
    parser.add_argument("--suffix", default='-USD.csv', help='Select only those files having input suffix.')

    preprocessing(parser.parse_args())
