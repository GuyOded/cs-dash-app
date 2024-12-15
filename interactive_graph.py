from dash import Dash, dcc, html, Input, Output, ctx
import plotly.express as px
import pathlib
import mca_output


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
            html.Div([html.Label("ROI"), html.Div(id="roi-out", children="[-infinity, infinity]")])
        ])

        self._app.run(debug=True)

    def _register_callbacks(self):
        @self._app.callback(Output("roi-out", "children"),
                Input("mca-output-graph", "selectedData"))
        def print_selected_region(selected_data):
            if not selected_data:
                return

            return f"[{selected_data["points"][0]["pointNumber"]}, {selected_data["points"][-1]["pointNumber"]}]"

        @self._app.callback(Output("mca-output-graph", "figure"),
                Input("fit", "n_clicks"),
                Input("mca-output-graph", "selectedData"),
                Input("coeff", "value"),
                Input("expect", "value"),
                Input("deviation", "value"))
        def generate_fit(_, selected_roi, coeff, expect, deviation):
            trigger_id = ctx.triggered_id
            if trigger_id != "fit":
                return
