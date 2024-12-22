import numpy as np

def find_background_noise(channel_counts: np.ndarray) -> tuple[float, float]:
    errors = np.sqrt(channel_counts)
    mean = np.average(channel_counts)

    mean_error = np.sqrt(np.sum(np.square(errors))) / len(channel_counts)

    return mean, mean_error
