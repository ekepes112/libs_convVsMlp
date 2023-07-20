RESULTS_PATH = '/content/gdrive/My Drive/projects/convVsMlp/temp'
LR_SCAN_END = 1e-1
LR_SCAN_START = 1e-8
LR_SCAN_STEP_SIZE = .2
LR_SCAN_WARMUP = 10
BATCH_SIZE = 64
CV_EPOCHS = 250

OPTIMIZERS = [
    'Adam_1',
    'Adam_2',
    'Adam_3',
    'Adam_4',
    'Adam_5',
    'Adam_6',
    'Adam_7',
    'Adam_8',
    'Adam_9',
    'SGD_1',
    'SGD_2',
    'SGD_3',
    'SGD_4',
    'SGD_5',
    'SGD_6',
    'SGD_7',
    'SGD_8',
]

OPTIMIZER_PARAMS = {
    'Adam_1': dict(
        name='Adam_1',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.9,
        beta_2=.999
    ),
    'Adam_2': dict(
        name='Adam_2',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.8,
        beta_2=.999
    ),
    'Adam_3': dict(
        name='Adam_3',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.7,
        beta_2=.999
    ),
    'Adam_4': dict(
        name='Adam_4',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.9,
        beta_2=.9
    ),
    'Adam_5': dict(
        name='Adam_5',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.8,
        beta_2=.9
    ),
    'Adam_6': dict(
        name='Adam_6',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.7,
        beta_2=.9
    ),
    'Adam_7': dict(
        name='Adam_7',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.9,
        beta_2=.8
    ),
    'Adam_8': dict(
        name='Adam_8',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.8,
        beta_2=.8
    ),
    'Adam_9': dict(
        name='Adam_9',
        optimizer_type='Adam',
        learning_rate=LR_SCAN_START,
        beta_1=.7,
        beta_2=.8
    ),
    'SGD_1': dict(
        name='SGD_1',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0,
        nesterov=False
    ),
    'SGD_2': dict(
        name='SGD_2',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.2,
        nesterov=False
    ),
    'SGD_3': dict(
        name='SGD_3',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.4,
        nesterov=False
    ),
    'SGD_4': dict(
        name='SGD_4',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.6,
        nesterov=False
    ),
    'SGD_5': dict(
        name='SGD_5',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0,
        nesterov=True
    ),
    'SGD_6': dict(
        name='SGD_6',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.2,
        nesterov=True
    ),
    'SGD_7': dict(
        name='SGD_7',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.4,
        nesterov=True
    ),
    'SGD_8': dict(
        name='SGD_8',
        optimizer_type='SGD',
        learning_rate=LR_SCAN_START,
        momentum=0.6,
        nesterov=True
    ),
}