from dash import Dash, dcc, html, Input, callback
import plotly.express as px
import pathlib
import mca_output


app = Dash()


def show_interactive_graph(output_file_path: pathlib.Path, mca_output: mca_output.MCAOutput):
    """
    Shows interactive graph
    """
    figure = px.scatter(y=mca_output.channel_count_list)
    figure.update_layout(dragmode="select",
                         selectdirection="h",
                         xaxis_title=None,
                         yaxis_title=None)

    app.layout = html.Div([
        html.H1(f"{output_file_path.stem}"),
        dcc.Graph("mca-output-graph", figure=figure)
    ])

    app.run(debug=True)

@callback(Input("mca-output-graph", "selectedData"))
def print_selected(selected_data):
    if not selected_data:
        return

    print(f"{selected_data["points"][:10]}...")
