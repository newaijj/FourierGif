import numpy as np


def normalize_points(points):
    mean = np.mean(points, axis=0)
    return points - mean


def points_to_complex(points):
    """
	Input: list of (x,y) points
	Output: list of complex numbers
	"""
    return np.array([np.complex(x, y) for x, y in list(points)])


def signal_to_coeff_bin(signal):
    """
	Input: list of complex numbers
	Output: list of (coefficient,frequency pairs)
	"""
    coeffs = np.fft.fft(signal) / signal.size
    freqs = np.fft.fftfreq(signal.size, d=1 / signal.size)
    return list(zip(coeffs, freqs))


test_points = np.array([[4, 7], [0, 3], [-2, -4], [-7, -9], [3, 6]])

rand_test = np.random.rand(500, 2)
"""
norm_points = normalize_points(rand_test)
norm_x = [x for x,_ in norm_points]
norm_y = [y for _,y in norm_points]
signal = points_to_complex(norm_points)
co_bi = signal_to_coeff_bin(signal)
"""


def points_to_coeff_bin(points):
    norm_points = normalize_points(points)
    norm_x = [x for x, _ in norm_points]
    norm_y = [y for _, y in norm_points]
    signal = points_to_complex(norm_points)
    co_bi = signal_to_coeff_bin(signal)
    return co_bi, norm_x, norm_y
