"""
Fetch data related to coins and store them into data folder
"""
import coinpy as cp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    cp.Collector().fetch_and_save(path=args.path)
