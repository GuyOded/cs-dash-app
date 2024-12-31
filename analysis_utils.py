import numpy as np
from mca_output import MCAOutput
from curve_fitter import ModelData, GaussianFittingParameters


def find_background_noise(channel_counts: np.ndarray) -> tuple[float, float]:
    errors = np.sqrt(channel_counts)
    mean = np.average(channel_counts)

    mean_error = np.sqrt(np.sum(np.square(errors))) / len(channel_counts)

    return mean, mean_error


def generate_model_data_from_mca_output(mca_output: MCAOutput, normalize_time=True) -> ModelData:
    channel_counts = np.array(mca_output.channel_count_list, dtype=np.float32)
    channel_count_resolution = 1
    if normalize_time:
        channel_counts /= mca_output.measurement_time
        channel_count_resolution = 1 / mca_output.measurement_time

    channel_count_uncertainty = np.sqrt(np.full([len(channel_counts)], channel_count_resolution / 3) + channel_counts / (mca_output.measurement_time**2))

    channel_index = np.arange(len(mca_output.channel_count_list), dtype=np.float32)
    channel_index_uncertainty = np.full([len(channel_index)], 1/np.sqrt(3))

    return ModelData(channel_index, channel_index_uncertainty, channel_counts, channel_count_uncertainty)

def slice_model_data(model_data: ModelData, start_index: int, end_index: int):
    return ModelData(model_data.x_data[start_index:end_index], model_data.x_error[start_index:end_index], model_data.y_data[start_index:end_index], model_data.y_error[start_index:end_index])


def print_guassian_sum_fit_output(fitting_datas: list[GaussianFittingParameters], uncertainties: list[float], chi_sq, p_value):
    for i, (gaussian_fit, uncertainty) in enumerate(zip(fitting_datas, uncertainties)):
        print(f"Gaussian #{i+1}: Fitting Params: {gaussian_fit}, Uncertainties: {uncertainty}")

    print(f"Chi sq: {chi_sq}, P-Value: {p_value}")
