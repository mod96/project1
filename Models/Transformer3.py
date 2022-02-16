from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, MultiHeadAttention, Conv1D,
                                     Add, LayerNormalization,
                                     Dropout, Embedding, Flatten,
                                     Layer)
from tensorflow.keras.activations import relu

from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW

from kerastuner import HyperModel
from keras_tuner import Hyperband

from settings import SEARCH_SAVE_DIRECTORY

from Callbacks import LRSchedule
"""
https://keras.io/examples/timeseries/timeseries_transformer_classification/
"""


class ViTEncoderHyperModel(HyperModel):
    def __init__(self, n_features, timelen, **kwargs):
        super(ViTEncoderHyperModel, self).__init__(**kwargs)
        self.n_features = n_features
        self.timelen = timelen

        # User choice part
        self.n_encoders = [8]
        self.n_heads = [12]
        self.att_dropouts = [0.2]

        self.ffn_nums = [4]
        self.ffn_dropouts = [0.2]

        self.fwd_nums = [4]
        self.fwd_dropouts = [0.2]

        self.lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        self.optimizer = ["Adam"]
        self.optimizer_func = [Adam]

    def build(self, hp):
        model = ViTEncoder(self.n_features, self.timelen,

                           n_encoders=hp.Choice("n_encoders", self.n_encoders),
                           n_heads=hp.Choice("n_heads", self.n_heads),
                           att_dropout=hp.Choice("att_dropout", self.att_dropouts),

                           ffn_num=hp.Choice("ffn_num", self.ffn_nums),
                           ffn_dropout=hp.Choice("ffn_dropout", self.ffn_dropouts),

                           fwd_num=hp.Choice("fwd_num", self.fwd_nums),
                           fwd_dropout=hp.Choice("fwd_dropout", self.fwd_dropouts)
                           )

        optimizer = hp.Choice("optimizer", self.optimizer)
        for func, opt in zip(self.optimizer_func, self.optimizer):
            if optimizer == opt:
                if opt == "AdamW":
                    with hp.conditional_scope("optimizer", [opt]):
                        model.compile(optimizer=func(weight_decay=0.01,
                                                     learning_rate=hp.Choice("lrs", self.lrs)),
                                      loss="MSE")
                else:
                    with hp.conditional_scope("optimizer", [opt]):
                        model.compile(optimizer=func(learning_rate=hp.Choice("lrs", self.lrs)),
                                      loss="MSE")
                break
        return model


def get_ViT_tuner3(n_features, timelen):
    hypermodel = ViTEncoderHyperModel(n_features=n_features, timelen=timelen)
    tuner = Hyperband(hypermodel,
                      objective='val_loss',
                      max_epochs=1,
                      directory=SEARCH_SAVE_DIRECTORY,
                      project_name='transformer3'
                      )
    return tuner


class ViTEncoder(Model, ABC):
    def __init__(self, n_features, timelen, n_encoders=3,
                 n_heads=8, embed_dim=32, att_dropout=0,
                 ffn_num=4, ffn_dropout=0.2, ffn_dim=128,
                 fwd_num=1, fwd_dropout=0, fwd_dim=128):
        super(ViTEncoder, self).__init__()
        self.PatchEmbedding = PatchEmbedding(timelen, embed_dim=embed_dim)
        self.Encoders = [Encoder(n_heads, embed_dim, att_dropout,
                                 ffn_num, ffn_dropout, ffn_dim) for _ in range(n_encoders)]
        self.Flatten = Flatten()
        self.MLP = MLP(n_features, fwd_dim, fwd_num, fwd_dropout)
        self.Add = Add()

    def call(self, inputs, training=None, mask=None):
        x = self.PatchEmbedding(inputs)
        for encoder in self.Encoders:
            x = encoder(x)
        x = self.Flatten(x)
        x = self.MLP(x)
        out = self.Add([inputs[:, 0, :], x])
        return out


# Layers for ViT
class PatchEmbedding(Layer):
    def __init__(self, timelen, embed_dim=32):
        super(PatchEmbedding, self).__init__()
        self.positions = tf.range(start=0, limit=timelen, delta=1)
        self.dense = Dense(embed_dim)
        self.embedding = Embedding(input_dim=timelen, output_dim=embed_dim)

    def call(self, inputs, *args, **kwargs):
        x1 = self.dense(inputs)  # (batch, timelen, embed_dim)
        x2 = self.embedding(self.positions)  # (timelen, embed_dim)
        return x1 + x2


class Encoder(Layer):
    def __init__(self, n_heads=8, embed_dim=32, att_dropout=0.,
                 ffn_num=1, ffn_dropout=0., ffn_dim=128):
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.DP1 = Dropout(att_dropout)
        self.LN1 = LayerNormalization(epsilon=1e-6)
        self.Add1 = Add()

        self.MLP = FFN(embed_dim, ffn_dim, ffn_layers=ffn_num, dropout=ffn_dropout)
        self.Add2 = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.MHA(inputs, inputs)
        x = self.DP1(x)
        x = self.LN1(x)
        added = self.Add1([x, inputs])
        x2 = self.MLP(added)
        out = self.Add2([x2, added])
        return out


class FFN(Layer):
    def __init__(self, embed_dim, ffn_dim, ffn_layers, dropout):
        super(FFN, self).__init__()
        assert ffn_layers > 0, "ffn_layers must be larger than 0"
        self.layers = [[Conv1D(filters=ffn_dim, kernel_size=1, activation=relu), Dropout(dropout)]
                       for _ in range(ffn_layers - 1)] + \
                      [[Conv1D(filters=embed_dim, kernel_size=1), LayerNormalization(epsilon=1e-6)]]

    def call(self, x, *args, **kwargs):
        for layers in self.layers:
            for layer in layers:
                x = layer(x)
        return x


class MLP(Layer):
    def __init__(self, embed_dim, fwd_dim, mlp_layers, dropout):
        super(MLP, self).__init__()
        self.layers = [[Dense(fwd_dim, activation=relu), Dropout(dropout)]
                       for _ in range(mlp_layers)]
        self.out = Dense(embed_dim)

    def call(self, x, *args, **kwargs):
        for layers in self.layers:
            for layer in layers:
                x = layer(x)
        out = self.out(x)
        return out


if __name__ == "__main__":
    model = ViTEncoder(60, 50)
    print(model.summary())
