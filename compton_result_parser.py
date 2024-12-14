import argparse
import pathlib


def main():
    pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compton Parser Utility", description="A utility to help parse and analyze results from MCA.")
    parser.add_argument("file", required=True, help="Path to an output file of MCA, or alternatively to a folder containing output files as immediate children.", type=pathlib.Path)
    parser.add_argument("-csv", "-gen-csv", action="store_true", type=bool)

    return parser


if __name__ == "__main__":
    main()
