import numpy as np
import scipy.optimize


def model_gaussian_fit(x: float, expectancy: float, deviation: float, coefficient: float):
    return (coefficient / np.sqrt(2 * np.pi * deviation**2)) * np.exp(-(x - expectancy)**2 / (2 * (deviation**2)))


def fit_gaussian_curve(xdata, ydata, guess=None, y_error=None) -> tuple[list, list]:
    """
    Fits a gaussian curve with least squares
    """
    return scipy.optimize.curve_fit(model_gaussian_fit, xdata, ydata, guess, y_error)
