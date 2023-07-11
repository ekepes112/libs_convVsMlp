from tensorflow.keras.optimizers import Adam, SGD
import config


def generate_optimizers() -> dict:
    return {
        'Adam_1': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.9,
            beta_2=.999
        ),
        'Adam_2': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.8,
            beta_2=.999
        ),
        'Adam_3': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.7,
            beta_2=.999
        ),
        'Adam_4': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.9,
            beta_2=.9
        ),
        'Adam_5': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.8,
            beta_2=.9
        ),
        'Adam_6': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.7,
            beta_2=.9
        ),
        'Adam_7': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.9,
            beta_2=.8
        ),
        'Adam_8': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.8,
            beta_2=.8
        ),
        'Adam_9': Adam(
            learning_rate=config.LR_SCAN_START,
            beta_1=.7,
            beta_2=.8
        ),
        'SGD_1': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0,
            nesterov=False
        ),
        'SGD_2': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.2,
            nesterov=False
        ),
        'SGD_3': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.4,
            nesterov=False
        ),
        'SGD_4': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.6,
            nesterov=False
        ),
        'SGD_5': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0,
            nesterov=True
        ),
        'SGD_6': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.2,
            nesterov=True
        ),
        'SGD_7': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.4,
            nesterov=True
        ),
        'SGD_8': SGD(
            learning_rate=config.LR_SCAN_START,
            momentum=0.6,
            nesterov=True
        ),
    }
