from tensorflow.keras.callbacks import Callback, EarlyStopping
from datetime import datetime


# TODO : figure out how to parse "hyper_parameters" and save it as - {model_name}_{hps}_{epoch}
class CheckpointSave(Callback):
    def __init__(self, model_name, term=20, model_num=0):
        super(CheckpointSave, self).__init__()
        self.model_name = model_name
        self.term = term
        self.model_num = model_num

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and epoch % self.term == 0:
            self.model.save_weights(self.model_dir + str(self.model_num) + '_' + str(epoch))
            print("weights saved to : ", self.model_dir + str(self.model_num) + '_' + str(epoch))


# TODO : same as above
class EarlyStopAndSave(EarlyStopping):
    def __init__(self, **kwargs):
        super(EarlyStopAndSave, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        super(EarlyStopAndSave, self).on_train_end(logs)
        # save method but ckpt_save and this can be duplicate so this name must be '_final'


"""
train 시 _final이 없으면 가장 마지막 weight load해서 다시 시작.
있으면 그냥 _final load함.
"""