from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, GRU, Dense

from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow_addons.optimizers import AdamW

from kerastuner import HyperModel
from keras_tuner import Hyperband

from settings import MAX_EPOCH, SEARCH_SAVE_DIRECTORY


class Baseline(Model, ABC):
    def __init__(self, n_features, hidden=128, num_layers=3):
        super(Baseline, self).__init__()
        self.rnn_list = [Bidirectional(GRU(hidden, return_sequences=True))
                         for _ in range(num_layers - 1)] + [Bidirectional(GRU(hidden))]
        self.fc = Dense(n_features)

    def call(self, x, training=None, mask=None):
        batched_first_values = x[:, 0, :]
        for layer in self.rnn_list:
            x = layer(x)
        x = tf.nn.relu(x)
        x = self.fc(x)
        return x + batched_first_values


class BaselineHyperModel(HyperModel):
    def __init__(self, n_features, **kwargs):
        super(BaselineHyperModel, self).__init__(**kwargs)
        self.n_features = n_features

        # User choice part
        self.hiddens = [32, 64, 128, 256]
        self.layers = [1, 2, 3, 4]
        self.lrs = [1e-2, 1e-3, 1e-4]
        self.optimizer = ["Adam", "Nadam", "AdamW"]
        self.optimizer_func = [Adam, Nadam, AdamW]

    def build(self, hp):
        hidden = hp.Choice("hidden", self.hiddens)
        num_layers = hp.Choice("layers", self.layers)
        model = Baseline(n_features=self.n_features, hidden=hidden, num_layers=num_layers)
        optimizer = hp.Choice("optimizer", self.optimizer)
        for func, opt in zip(self.optimizer_func, self.optimizer):
            if optimizer == opt:
                if opt == "AdamW":
                    with hp.conditional_scope("optimizer", [opt]):
                        model.compile(optimizer=func(weight_decay=0.01,
                                                     learning_rate=hp.Choice("lr", self.lrs)),
                                      loss="MSE")
                else:
                    with hp.conditional_scope("optimizer", [opt]):
                        model.compile(optimizer=func(learning_rate=hp.Choice("lr", self.lrs)),
                                      loss="MSE")
                break
        return model


def get_baseline_tuner(n_features):
    hypermodel = BaselineHyperModel(n_features=n_features)
    tuner = Hyperband(hypermodel,
                      objective='val_loss',
                      max_epochs=MAX_EPOCH//4,
                      directory=SEARCH_SAVE_DIRECTORY,
                      project_name='baseline'
                      )
    return tuner


if __name__ == "__main__":
    pass
