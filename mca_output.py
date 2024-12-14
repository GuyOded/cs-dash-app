import pathlib
from dataclasses import dataclass


_LIVE_TIME_MARKER = "LIVE_TIME"
_DATA_SECTION_START_MARKER = "<<DATA>>"
_DATA_SECTION_END_MARKER = "<<END>>"


@dataclass(frozen=True)
class MCAOutput:
    path: str
    measurement_time: float
    channel_count_list: list[int]


def parse_output_file(output_file: str) -> MCAOutput:
    path = pathlib.Path(output_file)
    if not path.exists():
        raise ValueError(f"Path {output_file} does not exist.")

    with open(output_file, "r") as output:
        is_data_section = False
        data_list = []
        for line in output.readlines():
            if line.startswith(_LIVE_TIME_MARKER):
                measurement_time_text = line.split("-")[-1].strip()
                measurement_time = float(measurement_time_text)

            if line.startswith(_DATA_SECTION_END_MARKER):
                break

            if line.startswith(_DATA_SECTION_START_MARKER):
                is_data_section = True
                continue

            if is_data_section:
                channel_count = int(line)
                data_list.append(channel_count)

    return MCAOutput(output_file, measurement_time, data_list)
