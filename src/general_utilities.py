from pathlib import Path

def check_dirs(path:Path) -> None:
    model_dir = path.joinpath('models')

    if not model_dir.exists():
        print('creating saved models directory')
        model_dir.mkdir()
    else:
        print('saved models directory already exists')

    checkpoint_dir = path.joinpath('checkpoints')

    if not checkpoint_dir.exists():
        print('creating checkpoint directory')
        checkpoint_dir.mkdir()
    else:
        print('checkpoint directory already exists')

