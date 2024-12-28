import typing
import numpy as np
import scipy.optimize
import scipy.odr as odr
import scipy.stats as stats
import itertools
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


def efficiency_model(beta: tuple[float, float], x: float):
    linear, squared = beta

    return (linear / x) + (squared / x**2)


def linear_model(beta: tuple[float, float], x: float):
    free_term = beta[0]
    slope = beta[1]

    return slope*x + free_term


def two_gaussian_sum_model(beta: tuple[float, float, float, float, float, float], x: float):
    mean1, deviation1, normalization1, mean2, deviation2, normalization2 = beta

    return gaussian_model([mean1, deviation1, normalization1], x) + gaussian_model([mean2, deviation2, normalization2], x)


def gaussian_sum_model(beta: list[float], x: float):
    if len(beta) % 3 != 0:
        raise ValueError(f"Incompatible parameter list, len(beta) == {len(beta)} which is not a multiple of 3")

    gaussian_params_list = [GaussianFittingParameters(beta[i], beta[i+1], beta[i+2]) for i in range(0, len(beta), 3)]
    return np.array([gaussian_model([gaussian_params.mean, gaussian_params.std_dev, gaussian_params.normalization], x) for gaussian_params in gaussian_params_list]).sum(axis=0)


def vectorized_line(free_term: float, slope: float):
    return np.vectorize(lambda x: linear_model([free_term, slope], x))


def vectorized_gaussian(mean, deviation, normalization):
    return np.vectorize(lambda x: gaussian_model([mean, deviation, normalization], x))


def vectorized_two_gaussian_sum(gaussian_parameters: tuple[float, float, float, float, float, float]):
    mean1, deviation1, normalization1, mean2, deviation2, normalization2 = gaussian_parameters
    return np.vectorize(lambda x: gaussian_model([mean1, deviation1, normalization1], x) + gaussian_model([mean2, deviation2, normalization2], x))


def vectorized_gaussian_sum(gaussian_parameters: list[GaussianFittingParameters]):
    flattened_params = list(itertools.chain.from_iterable([[params.mean, params.std_dev, params.normalization] for params in gaussian_parameters]))
    return np.vectorize(lambda x: gaussian_sum_model(flattened_params, x))


def vectorized_efficiency_model(fit_params: tuple[float, float]):
    return np.vectorize(lambda x: efficiency_model(fit_params, x))


def ols_fit_gaussian_curve(x_data, y_data, guess=None, y_error=None) -> tuple[list, list]:
    """
    Fits a gaussian curve with least squares
    """
    return scipy.optimize.curve_fit(gaussian_model, x_data, y_data, guess, y_error)


def odr_fit_gaussian(model_data: ModelData, guess: list[float]) -> tuple[list[float], list[float], float, float]:
    """
    Fits a gaussian curve with odr
    """
    return odr_fit(model_data, gaussian_model, guess)


def odr_fit_two_gaussian_sum(model_data: ModelData, guess: list[float]) -> tuple[list[float], list[float], float, float]:
    """
    Fits two gaussians using odr
    """
    return odr_fit(model_data, two_gaussian_sum_model, guess)


def odr_fit_gaussian_sum(model_data: ModelData, guess: list[GaussianFittingParameters]) -> tuple[list[GaussianFittingParameters], list[tuple[float, float, float]], float, float]:
    guesses = list(itertools.chain.from_iterable([[gaussian_parameters.mean, gaussian_parameters.std_dev, gaussian_parameters.normalization] for gaussian_parameters in guess]))
    beta, std_dev, chi_sq, p_value = odr_fit(model_data, gaussian_sum_model, guesses)

    fit_params_list = [GaussianFittingParameters(beta[i], beta[i+1], beta[i+2]) for i in range(0, len(beta), 3)]
    std_dev_list = [(std_dev[i], std_dev[i+1], std_dev[i+2]) for i in range(0, len(std_dev), 3)]

    return fit_params_list, std_dev_list, chi_sq, p_value


def odr_fit_efficiency(model_data: ModelData, guess: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    return odr_fit(model_data, efficiency_model, guess)


def odr_fit(model_data: ModelData, model_func: typing.Callable[[list[float], float], float], guess: list[float]) -> tuple[list[float], list[float], float, float]:
    model = odr.Model(model_func)
    data = odr.Data(model_data.x_data, model_data.y_data, wd=1/np.power(model_data.x_error, 2), we=1/np.power(model_data.y_error, 2))

    odr_runner = odr.ODR(data, model, beta0=guess)
    output = odr_runner.run()
    degrees_of_freedom = len(model_data.x_data) - len(guess)

    p_value = 1 - stats.chi2.cdf(output.res_var, degrees_of_freedom)

    return (output.beta, output.sd_beta, output.res_var, p_value)
