from dash import Dash, dcc, html, Input, Output, ctx, no_update
import plotly.express as px
import plotly.graph_objects as go
import pathlib
import mca_output
import numpy as np
import curve_fitter
import json
from dataclasses import dataclass, asdict


class InteractiveGraphInterface:
    def __init__(self, mca_output: mca_output.MCAOutput, output_path: pathlib.Path):
        self._app = Dash()
        self._output_file_path = output_path
        self._mca_output = mca_output
        self._register_callbacks()

    def show_interactive_graph(self):
        """
        Shows interactive graph
        """
        figure = px.scatter(y=self._mca_output.channel_count_list)
        figure.update_layout(dragmode="select",
                            selectdirection="h",
                            xaxis_title=None,
                            yaxis_title=None)

        json_config_file = self._output_file_path.parent.joinpath(f"{self._output_file_path.stem}.json")
        if json_config_file.exists():
            with open(json_config_file, "r") as json_config:
                loaded_data = json.load(json_config)
                curve_data = CurveData(**loaded_data)
            curve_y_data = self._calculate_gaussian_curve_values(curve_data.expectancy, curve_data.deviation, curve_data.coefficient)
            figure = self._show_curve_over_output_scatter(curve_y_data)

        self._app.layout = html.Div([
            html.H1(f"{self._output_file_path.stem}"),
            dcc.Graph("mca-output-graph", figure=figure),
            html.Div(children=[html.H3("Fit"),
                            html.Div([
                                html.Span([html.Label("Coefficient"), dcc.Input(None, type="number", id="coeff")]),
                                html.Br(),
                                html.Span([html.Label("Expectancy"), dcc.Input(None, type="number", id="expect")]),
                                html.Br(),
                                html.Span([html.Label("Deviation"), dcc.Input(None, type="number", id="deviation")])
                            ]),
                            html.Button(id="fit", n_clicks=0, children="fit")]),
            html.Div([html.Label("ROI"), html.Div(id="roi-out", children="[-infinity, infinity]")]),
            html.Div([html.Label("Suggested Fit Coefficient"), html.Div(id="approx-coeff", )])
        ])

        self._app.run(debug=True)

    def _register_callbacks(self):
        @self._app.callback(Output("roi-out", "children"),
                            Output("approx-coeff", "children"),
                            Input("mca-output-graph", "selectedData"))
        def print_selected_region(selected_data):
            if not selected_data or not selected_data["points"]:
                return "[-infinity, infinity]", "0"

            total_counts_under_curve = sum(self._mca_output.channel_count_list[selected_data["points"][0]["pointIndex"]:selected_data["points"][-1]["pointIndex"]])

            return f"[{selected_data["points"][0]["pointNumber"]}, {selected_data["points"][-1]["pointNumber"]}]", total_counts_under_curve

        @self._app.callback(Output("mca-output-graph", "figure"),
                Input("fit", "n_clicks"),
                Input("mca-output-graph", "selectedData"),
                Input("coeff", "value"),
                Input("expect", "value"),
                Input("deviation", "value"))
        def generate_fit(_, selected_roi, guess_coeff, guess_expect, guess_deviation):
            trigger_id = ctx.triggered_id
            if trigger_id != "fit":
                return no_update

            xdata = np.arange(len(self._mca_output.channel_count_list))
            ydata = np.array(self._mca_output.channel_count_list)

            if selected_roi and selected_roi["points"]:
                xdata = np.arange(selected_roi["points"][0]["pointIndex"], selected_roi["points"][-1]["pointIndex"])
                ydata = np.array(self._mca_output.channel_count_list[selected_roi["points"][0]["pointIndex"]:selected_roi["points"][-1]["pointIndex"]])

            y_error = np.sqrt(ydata)
            (params, params_error) = curve_fitter.fit_gaussian_curve(xdata, ydata, guess=[guess_expect, guess_deviation, guess_coeff], y_error=y_error)
            print(f"Param values: {params}\nParam errors: {params_error}")

            figure = px.scatter(y=self._mca_output.channel_count_list)
            figure.update_layout(dragmode="select",
                            selectdirection="h",
                            xaxis_title=None,
                            yaxis_title=None)

            expectancy = params[0]
            std_dev = params[1]
            coefficient = params[2]
            fitted_line_y = self._calculate_gaussian_curve_values(expectancy, std_dev, coefficient)

            self._save_fit_params_to_json(expectancy, std_dev, coefficient, params_error[0][0], params_error[1][1], params_error[2][2])

            return self._show_curve_over_output_scatter(fitted_line_y)

    def _show_curve_over_output_scatter(self, curve_y_data: np.ndarray) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self._mca_output.channel_count_list, mode="markers", name="MCA out"))
        fig.add_trace(go.Scatter(y=curve_y_data, mode="lines", name="Fitted Curve"))
        fig.update_layout(dragmode="select",
                        selectdirection="h",
                        xaxis_title=None,
                        yaxis_title=None)

        return fig

    def _calculate_gaussian_curve_values(self, expectancy: float, std_dev: float, coefficient: float):
        return np.array([curve_fitter.model_gaussian_fit(x, expectancy, std_dev, coefficient) for x in range(len(self._mca_output.channel_count_list))])

    def _save_fit_params_to_json(self, expectancy, deviation, coefficient, expectancy_error, deviation_error, coefficient_error):
        curve_y_values = self._calculate_gaussian_curve_values(expectancy, deviation, coefficient)

        # TODO: Add error for integral
        counts_integral = sum(curve_y_values)
        curve_data = CurveData(expectancy, expectancy_error, deviation, deviation_error, coefficient, coefficient_error, counts_integral)

        curve_data_dictionary = asdict(curve_data)
        with open(self._output_file_path.parent.joinpath(f"{self._output_file_path.stem}.json"), "w+") as curve_data_file:
            json.dump(curve_data_dictionary, curve_data_file, indent=4)


@dataclass(frozen=True)
class CurveData:
    expectancy: float
    expectancy_error: float
    deviation: float
    deviation_error: float
    coefficient: float
    coefficient_error: float
    counts_integral: float
