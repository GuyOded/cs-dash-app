import argparse
import pathlib
import mca_output


def main():
    parser = build_argparser()
    args = parser.parse_args()
    output_file: pathlib.Path = args.file

    if output_file.is_dir():
        raise NotImplemented("This part is not implemented yet")

    output_data = mca_output.parse_output_file(output_file)
    if args.gen_csv:
        build_csv(output_file, output_data, args.normalize_time)


def build_csv(output_file_path: pathlib.Path, output_data: mca_output.MCAOutput, normalize_time=False):
    output_csv_path = output_file_path.resolve().parent.joinpath(f"{output_file_path.stem}.csv")
    with open(output_csv_path, "w") as csv_file:
        csv_file.write("Channel,Count\n")
        for i, count in enumerate(output_data.channel_count_list):
            if normalize_time:
                count /= output_data.measurement_time

            csv_file.write(f"{i},{count}\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compton Parser Utility", description="A utility to help parse and analyze results from MCA.")
    parser.add_argument("file", help="Path to an output file of MCA, or alternatively to a folder containing output files as immediate children.", type=pathlib.Path)
    parser.add_argument("-csv", "--gen-csv", action="store_true", dest="gen_csv")
    parser.add_argument("-dt", "--normalize-time", action="store_true", dest="normalize_time", default=False)

    return parser


if __name__ == "__main__":
    main()
