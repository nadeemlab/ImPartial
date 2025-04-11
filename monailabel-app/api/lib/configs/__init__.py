from Impartial_classes import ImPartialConfig


class Vectra2Ch(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=8,
            n_channels=2,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]},
                '1': {'classes': 2, 'rec_channels': [1], 'ncomponents': [1, 1, 2]}
            }
        )


class Vectra2Ch1task(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=8,
            n_channels=2,
            npatches_epoch=128,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0,1], 'ncomponents': [2, 2]}
            }
        )


class DAPI1CH(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            EPOCHS=2,
            BATCH_SIZE=8,
            n_channels=1,
            npatches_epoch=128,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]}
            }
        )


class Config_CH3(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=64,
            n_channels=3,
            classification_tasks={
                # '0': {'classes': 1, 'rec_channels': [0,1,2], 'ncomponents': [2, 2]}
                '0': {'classes': 1, 'rec_channels': [0,1,2], 'ncomponents': [3, 3]}
            }
        )


class Config_CH2(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=64,
            n_channels=2,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0,1], 'ncomponents': [2, 2]}
            }
        )


class Config_CH1(ImPartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=8,
            n_channels=1,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]}
            }
        )
