import plotly.graph_objects as go


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
