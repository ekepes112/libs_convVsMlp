import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from keras.engine import functional
from keras import models, optimizers
from tensorflow.math import exp
from keras.callbacks import LearningRateScheduler
from IPython.display import HTML, display
from graph_utils import _update_layout


def guess_learning_rate(fit_history: pd.DataFrame) -> float:
    tresholded_losses = fit_history.loc[
        fit_history.loc[:, 'loss'] < (
            fit_history.loc[:, 'loss'].iloc[:10].max() * 0.9
        ),
        :
    ]

    return (
        tresholded_losses.iloc[
            np.argmax(
                np.diff(
                    np.sign(
                        np.gradient(
                            np.gradient(tresholded_losses.loc[:, 'loss'])
                        )
                    )
                )
            ),
            :
        ].loc['lr']
    )


def estimate_learnig_rate(
    base_model: functional.Functional,
    tried_optimizers: dict,
    batch_size: int,
    train_data: tuple,
    results_path: Path,
    end_lr: float = 1e-1,
    step_size: float = .5,
    warmup_count: int = 10,
    clone_model: bool = True,
    overwrite_existing: bool = False,
    return_data: bool = False,
    training_verbosity: int = 0,
    save_fig: bool = True
):

    if not results_path.joinpath('lr_estimates').is_dir():
        results_path.joinpath('lr_estimates').mkdir()

    def scheduler(epoch, lr):
        if epoch < warmup_count:
            return (lr)
        else:
            return (lr * exp(step_size))

    for optimizer_name in tried_optimizers:
        save_path = results_path.joinpath(
            # f'lr_estimates/{base_model.name}_{optimizer_name}.html'
            {base_model.name}_{optimizer_name}.html'
        )
        if save_path.exists() and not overwrite_existing:
            print(f'Loading in training history for {optimizer_name}')
            display(HTML(filename=save_path))
            return (None)

        print(f'Estimating {optimizer_name}')

        if clone_model:
            model = models.clone_model(base_model)
        else:
            model = base_model

        init_lr = tried_optimizers.get(optimizer_name).learning_rate
        epoch_count = int(
            np.ceil(np.log(end_lr / init_lr)/step_size)
        ) + warmup_count

        model.compile(
            optimizer=tried_optimizers.get(optimizer_name),
            loss='mse'
        )

        training_history = model.fit(
            x=train_data[0],
            y=train_data[1],
            epochs=epoch_count,
            batch_size=batch_size,
            callbacks=[
                LearningRateScheduler(scheduler)
            ],
            verbose=training_verbosity
        )

        plot_data = pd.DataFrame(
            filter(
                lambda x: not np.isnan(x[1]),
                zip(
                    training_history.history.get('lr'),
                    training_history.history.get('loss')
                )
            ),
            columns=['lr', 'loss']
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_data['lr'],
                y=plot_data['loss']
            )
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        fig.update_layout(
            title=f'Model: {model.name}; Optimizer: {optimizer_name}',
            font=dict(
                family="Courier New, monospace",
                size=18
            )
        )

        fig = _update_layout(
            fig,
            x="Learning rate",
            y="Loss"
        )
        fig.show()
        if save_fig:
            fig.write_html(save_path)

        try:
            lr_guess = guess_learning_rate(plot_data)
        except:
            lr_guess = np.nan

        with open(save_path.with_suffix('.txt'), 'w') as file:
            file.write(f'{lr_guess}')

        if return_data:
            return (plot_data)
