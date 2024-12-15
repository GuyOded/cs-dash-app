import argparse
import pathlib
import mca_output
import plotly.graph_objects as go
import numpy as np
import interactive_graph


def main():
    parser = build_argparser()
    args = parser.parse_args()
    output_file: pathlib.Path = args.file

    if output_file.is_dir():
        raise NotImplemented("This part is not implemented yet")

    output_data = mca_output.parse_output_file(output_file)
    if args.gen_csv:
        build_csv(output_file, output_data, args.normalize_time)

    if args.show_graph:
        show_graph(output_file, output_data, args.normalize_time)

    if args.interactive:
        normalized_data = output_data
        if args.normalize_time:
            normalized_data = mca_output.MCAOutput(output_data.path, output_data.measurement_time, np.array(output_data.channel_count_list) / output_data.measurement_time)

        graph_interface = interactive_graph.InteractiveGraphInterface(normalized_data, output_file)
        graph_interface.show_interactive_graph()


def build_csv(output_file_path: pathlib.Path, output_data: mca_output.MCAOutput, normalize_time=False):
    output_csv_path = output_file_path.resolve().parent.joinpath(f"{output_file_path.stem}.csv")
    with open(output_csv_path, "w") as csv_file:
        csv_file.write("Channel,Count\n")
        for i, count in enumerate(output_data.channel_count_list):
            if normalize_time:
                count /= output_data.measurement_time

            csv_file.write(f"{i},{count}\n")


def show_graph(output_file_path: pathlib.Path, output_data: mca_output.MCAOutput, normalize_time=False):
    channel_count_list = np.array(output_data.channel_count_list)
    if normalize_time:
        channel_count_list /= output_data.measurement_time

    fig = go.Figure(
        data=[go.Bar(y=channel_count_list)],
        layout_title_text=f"Count Graph {output_file_path.stem}"
    )
    fig.show()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compton Parser Utility", description="A utility to help parse and analyze results from MCA.")
    parser.add_argument("file", help="Path to an output file of MCA, or alternatively to a folder containing output files as immediate children.", type=pathlib.Path)
    parser.add_argument("-csv", "--gen-csv", action="store_true", dest="gen_csv", help="Generates csv with columns: Channel, Count from the given output file.", default=False)
    parser.add_argument("-dt", "--normalize-time", action="store_true", dest="normalize_time", default=False, help="If true, all output counts will be normalized by live time.")
    parser.add_argument("-g", "--show-graph", action="store_true", dest="show_graph", default=False, help="Shows count-channel graph for a given file.")
    parser.add_argument("-i", "--interactive", action="store_true", dest="interactive", default=False, help="Opens measurement in interactive mode, let's you fit gaussians, select regions and so on")

    return parser


if __name__ == "__main__":
    main()
