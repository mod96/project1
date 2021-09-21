from tensorflow.keras.callbacks import Callback
import IPython


class ClearTrainingOutput(Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)
