import numpy as np


def range_check(series, size):
    data = []
    mean, check_std = 0, 0

    check_std = np.std(series[:size])
    for i in range(size, len(series)):
        std = np.std(series[i - size:i])
        mean = np.mean(series[i - size:i])
        max = np.max(series[i - size:i])

        if check_std * 2 >= std:
            check_std = std
            data.append(mean)
        elif max == series[i]:
            data.append(max * 5)
            check_std = std
        else:
            data.append(series[i] * 3)

    for _ in range(size):
        data.append(mean)

    return np.array(data)
