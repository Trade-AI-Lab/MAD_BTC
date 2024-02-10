import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
from typing import Union

def multiple_signals_same_graph(X:Union[list,np.array],dict_Y:dict):
    """
    Plot multiple signals on a same graph
    Args:
        X (list or np.array): list representing the axis X
        dict_Y (dict): dictionnary containing list of signals

    Returns:
        None , the figure is displayed in your browser
    """

    fig = go.Figure()
    for keys in list(dict_Y):
        fig.add_trace(go.Scatter(x=X,
                             y=dict_Y[keys],
                             mode='lines',
                             name=str(keys)))
    fig.show()

def subplots_plotly(X:Union[list,np.array],dict_Y:dict):
    """
    Plotting multiple signals on different graph with the same X axis

    Args:
        X (list or np.array): list representing the axis X
        dict_Y (dict): dictionnary containing list of signals

    Returns:

    """

    # Create a subplot grid
    fig = sp.make_subplots(rows=len(dict_Y), cols=1, shared_xaxes=True, vertical_spacing=0.05)

    n = 1
    for keys in list(dict_Y):
        fig.add_trace(go.Scatter(x=X, y=dict_Y[keys], mode='lines', name=keys, showlegend=False), row=n , col=1)
        n+=1

    # Update the layout
    fig.update_layout(
        title="Signals",
        xaxis=dict(title="Time"),
        height=900,
    )

    # Show the interactive plot
    fig.show()
def subplots_plotly_markers(X,dictionnary):
    import plotly.graph_objects as go
    fig = sp.make_subplots(rows=len(dictionnary), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    n = 1
    for key in dictionnary.keys():
        signal = dictionnary[key]["signal"]
        markers = dictionnary[key]["markers"]

        line_signal = go.Scatter(x=X,
            y=signal,
            name=f'{key}_signal',
            mode="lines")
        if markers is not None:
            marker_signal = go.Scatter(
                x=X,
                y=signal,
                name=f'{key}_markers',
                marker=dict(
                    color=markers,
                    colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                    size=8
                ),
                mode="markers",
                customdata=markers,
                hovertemplate=" highlight_signal= %{customdata} <br> %{y} <br> %{x}")
            fig.add_trace(marker_signal, row=n, col=1)
        fig.add_trace(line_signal, row=n, col=1)

        n += 1

        # Update the layout
        fig.update_layout(
            title="Signals",
            xaxis=dict(title="Time"),
            height=900,
        )

        # Show the interactive plot
    fig.show()
def threed_plot(X,Y,Z,x_title,y_title,z_title):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis=dict(title=x_title),
            yaxis=dict(title=y_title),
            zaxis=dict(title=z_title)
        )
    )

    # Show plot
    fig.show()


