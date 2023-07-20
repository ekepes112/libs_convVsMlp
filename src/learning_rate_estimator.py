import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gc
from pathlib import Path

from keras import models, optimizers
from keras.engine import functional
from keras.callbacks import LearningRateScheduler
from tensorflow.math import exp
from tensorflow.keras.backend import clear_session
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
    optimizer: optimizers.Optimizer,
    batch_size: int,
    train_data: tuple,
    results_path: Path,
    loss_function: str,
    end_lr: float = 1e-1,
    step_size: float = .5,
    warmup_count: int = 10,
    clone_model: bool = True,
    overwrite_existing: bool = False,
    return_data: bool = False,
    training_verbosity: int = 0,
    save_fig: bool = True,
):
    # prepare saving the results
    optimizer_name = optimizer.name
    print(f'Estimating {optimizer_name}')
    results_path = results_path.joinpath(f'lr_estimates/{base_model.name}')
    if not results_path.is_dir():
        results_path.mkdir(parents=True)
    save_path = results_path.joinpath(optimizer_name)
    if save_path.exists() and not overwrite_existing:
        print(f'Loading in training history for {optimizer_name}')
        display(HTML(filename=save_path.with_suffix('.html')))
        return (None)
    # prepare the lr scheduler
    def scheduler(epoch, lr):
        if epoch < warmup_count:
            return (lr)
        else:
            return (lr * exp(step_size))
    # prepare the model
    if clone_model:
        model = models.clone_model(base_model)
    else:
        model = base_model
    model.compile(
        optimizer=optimizer,
        loss=loss_function
    )
    # prepare the training
    init_lr = optimizer.learning_rate
    epoch_count = int(
        np.ceil(np.log(end_lr / init_lr)/step_size)
    ) + warmup_count
    # train the model
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
    # plot the learning curves
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
    # save the results
    if save_fig:
        fig.write_html(save_path.with_suffix('.html'))

    try:
        lr_guess = guess_learning_rate(plot_data)
    except:
        lr_guess = 3e-4

    with open(save_path.with_suffix('.txt'), 'w') as file:
        file.write(f'{lr_guess}')

    if return_data:
        return (plot_data)
    # clean up
    del model
    del optimizer
    clear_session()
    gc.collect()
