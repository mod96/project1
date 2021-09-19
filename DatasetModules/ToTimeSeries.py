from abc import ABC

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class HAIDataLoader(tf.data.Dataset, ABC):
    def __new__(cls, np_data_list, length=50, stride=3,
                batch_size=32, train=True):
        # for np_data in np_data_list:
        #     assert len(np_data.shape) == 2, 'np_data must be 2D'

        if train:
            seq_data_list = [TimeseriesGenerator(data=elt, targets=elt, length=length,
                                                 stride=stride, batch_size=batch_size,
                                                 shuffle=True)
                             for elt in np_data_list]
        else:
            seq_data_list = [TimeseriesGenerator(data=elt, targets=elt, length=length,
                                                 stride=stride, batch_size=batch_size)
                             for elt in np_data_list]

        def gen():
            for seq_data in seq_data_list:
                for x, y in seq_data:
                    yield x, y

        std_x_shape, std_y_shape = seq_data_list[0][0][0].shape, seq_data_list[0][0][1].shape
        # print(f'x_shape: {std_x_shape}   y_shape: {std_y_shape}')
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=std_x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=std_y_shape, dtype=tf.float32)
            )
        )


if __name__ == "__main__":
    import time
    from DatasetModules import load_dataset

    data_loading_start = time.perf_counter()
    np_data_list, _, _ = load_dataset('../datasets/train/')
    dataset = HAIDataLoader(np_data_list)  # .prefetch(100)
    print("Data loading time:", time.perf_counter() - data_loading_start)
    print("#"*50)

    start_time = time.perf_counter()
    for epoch_num in range(10):
        idx = 0
        epoch_start_time = time.perf_counter()
        for sample in dataset:
            idx += 1
        print(f"epoch {epoch_num + 1} execution time:", time.perf_counter() - epoch_start_time, "data count : ", idx)
    print("#" * 50)
    print("Total execution time:", time.perf_counter() - start_time)