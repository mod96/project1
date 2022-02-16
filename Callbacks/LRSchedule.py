import tensorflow as tf


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_state=2e-4, warm_up=20):
        self.initial_state = initial_state
        self.warm_up = warm_up

    def __call__(self, step):
        inc = self.initial_state * (step / self.warm_up)
        dec = self.initial_state * (step / self.warm_up) ** (-1)
        return tf.math.minimum(inc, dec)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    temp_learning_rate_schedule = LRSchedule()
    plt.rcParams['figure.figsize'] = [15, 15]
    plt.plot(temp_learning_rate_schedule(tf.range(200, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()

