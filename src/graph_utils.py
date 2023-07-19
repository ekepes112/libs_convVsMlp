import plotly.graph_objects as go
import matplotlib.pyplot as plt

CM = 1/2.54
AXIS_LABEL_TEXT_SIZE = 8
GRID_ALPHA = .1
MARKER_ALPHA = .5


def _update_layout(
    figure: go.Figure,
    x: str = 'Wavelength (nm)',
    y: str = 'Intensity (-)'
) -> go.Figure:
    return (
        figure.update_layout(
            legend=dict(
                x=.99,
                y=.95,
                yanchor="top",
                xanchor="right"
            ),
            margin=dict(
                t=50,
                b=60,
                l=60,
                r=10
            ),
            xaxis=dict(
                title=x,
                linecolor='rgba(25,25,25,.4)',
                mirror=True,
                linewidth=2,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(77,77,77,.1)'
            ),
            yaxis=dict(
                title=y,
                linecolor='rgba(25,25,25,.4)',
                mirror=True,
                linewidth=2,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(77,77,77,.1)'
            ),
            plot_bgcolor="#FFF"
        )
    )


def set_pyplot_defaults(save_fig: bool = False) -> None:
    plt.rcParams["figure.figsize"] = (
        16 * CM, 12 * CM) if save_fig else (16, 12)
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.titleweight'] = 1
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['grid.alpha'] = .1
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['lines.linewidth'] = .35
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['savefig.pad_inches'] = 0
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.format'] = 'tiff'
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['legend.markerscale'] = 3
    plt.rcParams['lines.markersize'] = .4
