from tensorflow.keras.callbacks import Callback, EarlyStopping

from settings import MODEL_SAVE_FOLDER


class CheckpointSave(Callback):
    def __init__(self, model_name, term=20):
        super(CheckpointSave, self).__init__()
        self.model_name = model_name
        self.term = term

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and epoch % self.term == 0:
            self.model.save_weights(f'{MODEL_SAVE_FOLDER}/{self.model_name}/{epoch}')
            print(f'Checkpoint Save : model saved to {MODEL_SAVE_FOLDER}/{self.model_name}/{epoch}')


class EarlyStopAndSave(EarlyStopping):
    def __init__(self, model_name, **kwargs):
        super(EarlyStopAndSave, self).__init__(**kwargs)
        self.model_name = model_name

    def on_train_end(self, logs=None):
        super(EarlyStopAndSave, self).on_train_end(logs)
        self.model.save_weights(f'{MODEL_SAVE_FOLDER}/{self.model_name}/final')
        print(f'EarlyStopAndSave : model saved to {MODEL_SAVE_FOLDER}/{self.model_name}/final')
