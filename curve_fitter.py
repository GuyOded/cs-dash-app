import typing
import numpy as np
import scipy.optimize
import scipy.odr as odr
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelData:
    x_data: np.typing.NDArray[np.float32]
    x_error: np.typing.NDArray[np.float32]
    y_data: np.typing.NDArray[np.float32]
    y_error: np.typing.NDArray[np.float32]


@dataclass(frozen=True)
class GaussianFittingParameters:
    mean: float
    std_dev: float
    normalization: float


def gaussian_model(beta: tuple[float, float, float], x: float):
    mean, deviation, normalization = beta
    return (normalization / np.sqrt(2 * np.pi * deviation**2)) * np.exp(-(x - mean)**2 / (2 * (deviation**2)))


def two_gaussian_sum_model(beta: tuple[float, float, float, float, float, float], x: float):
    mean1, deviation1, normalization1, mean2, deviation2, normalization2 = beta

    return gaussian_model([mean1, deviation1, normalization1], x) + gaussian_model([mean2, deviation2, normalization2], x)


def vectorized_gaussian(mean, deviation, normalization):
    return np.vectorize(lambda x: gaussian_model([mean, deviation, normalization], x))


def vectorized_two_gaussian_sum(gaussians_parameters: tuple[float, float, float, float, float, float]):
    mean1, deviation1, normalization1, mean2, deviation2, normalization2 = gaussians_parameters
    return np.vectorize(lambda x: gaussian_model([mean1, deviation1, normalization1], x) + gaussian_model([mean2, deviation2, normalization2], x))


def ols_fit_gaussian_curve(x_data, y_data, guess=None, y_error=None) -> tuple[list, list]:
    """
    Fits a gaussian curve with least squares
    """
    return scipy.optimize.curve_fit(gaussian_model, x_data, y_data, guess, y_error)


def odr_fit_gaussian(model_data: ModelData, guess: list[float]):
    """
    Fits a gaussian curve with odr
    """
    return odr_fit(model_data, gaussian_model, guess)


def odr_fit_gaussian_sum(model_data: ModelData, guess: list[float]):
    """
    Fits two gaussians using odr
    """
    return odr_fit(model_data, two_gaussian_sum_model, guess)


def odr_fit(model_data: ModelData, model_func: typing.Callable[[list[float], float], float], guess: list[float]):
    model = odr.Model(model_func)
    data = odr.Data(model_data.x_data, model_data.y_data, wd=1/np.power(model_data.x_error, 2), we=1/np.power(model_data.y_error, 2))

    odr_runner = odr.ODR(data, model, beta0=guess)
    output = odr_runner.run()

    # TOOD: Add p-value
    return (output.beta, output.sd_beta, output.res_var)
