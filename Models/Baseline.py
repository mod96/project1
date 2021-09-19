import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GRU, Dense

import keras_tuner as kt
from keras_tuner import Hyperband

class Baseline(tf.keras.Model):
    def __init__(self, n_features, hidden=100, num_layers=3):
        super(Baseline, self).__init__()
        self.rnn_list = [Bidirectional(GRU(hidden, return_sequences=True))
                         for _ in range(num_layers - 1)] + [Bidirectional(GRU(hidden))]
        self.fc = Dense(n_features)

    def call(self, x):
        batched_first_values = x[:, 0, :]
        for layer in self.rnn_list:
            x = layer(x)
        x = tf.nn.relu(x)
        x = self.fc(x)
        return x + batched_first_values

"""
class 사용법 : https://ichi.pro/ko/keras-tuner-mich-hiplot-eul-sayonghan-singyeongmang-haipeo-palamiteo-tyuning-129980331205114
함수형 : https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko
kt 가 알아서 저장을 해주는듯? => kt.Hyperband의 'directory'를 건드려주면 되는 것 같은데?
                        => model을 class로 구현해버리고 builder에서는 epoch만 받고, directory는 settings.py에서.
                        => callback은 만들긴 해야하는듯 (search해서 best_hps 찾고 그거 가져와서 다시 fit하기 때문)
"""
def make_builder(n_features):
    def model_builder(hp):
        hp_hidden = hp.Int
        hp_num_layers = hp.Int
        hp.Choice('units', [8, 16, 32])
        model = Baseline(n_features)


kt.Hyperband