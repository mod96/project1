from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, MultiHeadAttention,
                                     Add, LayerNormalization,
                                     Dropout, Embedding, Flatten,
                                     Layer)
from tensorflow.keras.activations import gelu

from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow_addons.optimizers import AdamW

from kerastuner import HyperModel
from keras_tuner import Hyperband

from settings import MAX_EPOCH, SEARCH_SAVE_DIRECTORY
"""
ViT paper: https://arxiv.org/pdf/2010.11929.pdf
keras simplified: https://keras.io/examples/vision/image_classification_with_vision_transformer/
"""


class ViTEncoder(Model, ABC):
    def __init__(self, n_features, timelen, n_encoders=3,
                 n_heads=8, embed_dim=32, att_dropout=0,
                 fwd_num=1, fwd_dropout=0, fwd_dim=128):
        super(ViTEncoder, self).__init__()
        self.PatchEmbedding = PatchEmbedding(timelen, embed_dim=embed_dim)
        self.Encoders = [Encoder(n_heads, embed_dim, att_dropout,
                                 fwd_num, fwd_dropout, fwd_dim) for _ in range(n_encoders)]
        self.Flatten = Flatten()
        self.Dense = Dense(n_features, activation='linear')
        self.Add = Add()

    def call(self, inputs, training=None, mask=None):
        x = self.PatchEmbedding(inputs)
        for encoder in self.Encoders:
            x = encoder(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        out = self.Add([inputs[:, -1, :], x])  # possible : 0 instead of -1
        return out


class ViTEncoderHyperModel(HyperModel):
    def __init__(self, n_features, timelen, **kwargs):
        super(ViTEncoderHyperModel, self).__init__(**kwargs)
        self.n_features = n_features
        self.timelen = timelen

        # User choice part
        self.n_encoders = [3, 4]
        self.n_heads = [8, 16]
        self.att_dropouts = [0.2, 0.4]
        self.fwd_nums = [3, 4]
        self.fwd_dropouts = [0.2, 0.4]

        self.lrs = [1e-3, 1e-4]
        self.optimizer = ["Adam", "Nadam", "AdamW"]
        self.optimizer_func = [Adam, Nadam, AdamW]

    def build(self, hp):
        model = ViTEncoder(self.n_features, self.timelen,
                           n_encoders=hp.Choice("n_encoders", self.n_encoders),
                           n_heads=hp.Choice("n_heads", self.n_heads),
                           att_dropout=hp.Choice("att_dropout", self.att_dropouts),
                           fwd_num=hp.Choice("fwd_num", self.fwd_nums),
                           fwd_dropout=hp.Choice("fwd_dropout", self.fwd_dropouts)
                           )

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


def get_ViT_tuner(n_features, timelen):
    hypermodel = ViTEncoderHyperModel(n_features=n_features, timelen=timelen)
    tuner = Hyperband(hypermodel,
                      objective='val_loss',
                      max_epochs=MAX_EPOCH//2,
                      directory=SEARCH_SAVE_DIRECTORY,
                      project_name='transformer'
                      )
    return tuner


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
    def __init__(self, n_heads=8, embed_dim=32, att_dropout=0,
                 fwd_num=1, fwd_dropout=0, fwd_dim=128):
        super(Encoder, self).__init__()
        self.LN1 = LayerNormalization()
        self.MHA = MultiHeadAttention(n_heads, embed_dim, dropout=att_dropout)
        self.Add1 = Add()
        self.LN2 = LayerNormalization()
        self.MLP = MLP(embed_dim, fwd_dim, mlp_layers=fwd_num, dropout=fwd_dropout)
        self.Add2 = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.LN1(inputs)
        x = self.MHA(x, x)
        added = self.Add1([x, inputs])
        x2 = self.LN2(added)
        x2 = self.MLP(x2)
        out = self.Add2([x2, added])
        return out


class MLP(Layer):
    def __init__(self, embed_dim, fwd_dim, mlp_layers, dropout):
        super(MLP, self).__init__()
        self.layers = [[Dense(fwd_dim, activation=gelu), Dropout(dropout)]
                       for _ in range(mlp_layers)]
        self.out = Dense(embed_dim)

    def call(self, x, *args, **kwargs):
        for layers in self.layers:
            for layer in layers:
                x = layer(x)
        out = self.out(x)
        return out


if __name__ == "__main__":
    model = tf.keras.Sequential(layers=[tf.keras.Input(shape=(50, 63)),
                                        PatchEmbedding(50),
                                        Encoder(),
                                        Flatten(),
                                        Dense(63)])
    print(model.summary())
