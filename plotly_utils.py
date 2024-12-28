import plotly.graph_objects as go
import numpy as np
from mca_output import MCAOutput


def generate_mca_out_figure(mca_out: MCAOutput, title: str, normalize_time=True) -> go.Figure:
    channel_count = np.array(mca_out.channel_count_list, dtype=float)
    if normalize_time:
        channel_count /= mca_out.measurement_time

    fig = go.Figure(data=go.Bar(y=channel_count))
    fig.update_layout(xaxis_title=None, yaxis_title=None, title=title)
    return fig


def generate_scatter_and_line_plot(scatter: np.ndarray, *lines: list[np.ndarray]):
    scatter = go.Scatter(y=scatter, mode="markers")
    lines_graph = [go.Scatter(y=line, mode="lines") for line in lines]
    fig = go.Figure(data=[scatter] + lines_graph)

    return fig

def     generate_scatter_with_x_and_lines_plot(scatter: np.ndarray, scatter_x: np.ndarray, *lines: list[np.ndarray], error_y = None):
    scatter = go.Scatter(y=scatter, x=scatter_x, mode="markers", error_y=dict(type="data", array=error_y, visible=True))
    lines_graph = [go.Scatter(y=line, x=scatter_x, mode="lines") for line in lines]
    fig = go.Figure(data=[scatter] + lines_graph)

    return fig
