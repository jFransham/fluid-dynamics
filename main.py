import cv2
import math
import matplotlib.pyplot as plot
import numpy as np

from itertools import chain

# Corresponds to dx and dy for finite-difference approximation
space_granularity = 0.001
time_granularity  = 0.01

# Kinematic viscosity of water at room temperature
viscosity = 8e-7

# This is just for readability, increasing it doesn't work because of indexing
dim = 2

grid_size = math.floor(1. / space_granularity)

grid_shape = [grid_size] * dim
# Shape for the pressure matrix
pre_shape = list(chain(grid_shape, [1]))
# Shape for the velocity matrix
vel_shape = list(chain(grid_shape, [dim]))

def get_last_next(field, axis):
    lst = np.roll(field, 1, axis)
    nxt = np.roll(field, -1, axis)

    return (lst, nxt)


def reduce(fn, itr, initial):
    tot = initial
    for i in itr:
        tot = fn(tot, i)
    return tot


def mul(a, b):
    return a * b


def product(itr):
    return reduce(mul, itr, 1)


def weighted_rms(result, guess):
    """
    Root mean difference squared, weighted by the average value
    """

    # Should be identical for `guess` and `result`
    n = product(np.shape(result))

    sum_result = np.sum(result)

    error_mat = result - guess
    # Flatten the matrix, so it is iterated as a list of numbers
    # instead of a list of lists
    flat_errors = np.reshape(error_mat, (n,))

    # Confusingly, this squares the elements, instead of multiplying the
    # matrix by itself
    sum_diff_square = np.sum(flat_errors ** 2)

    # This is equivalent to the RMS divided by the mean of `result`
    w_rms = math.sqrt(n * sum_diff_square) / sum_result

    return w_rms


def scalar_to_vec(*args):
    """
    Converts set of 2D scalar matrices to a matrix of vectors
    """
    list_a = list(args)

    if len(list_a[0].shape) <= 2:
        list_a = list(
            map(
                lambda arr: np.reshape(
                    arr,
                    list(
                        chain(arr.shape, [1])
                    )
                ),
                args
            )
        )

    return np.concatenate(list_a, axis=2)


def gradient_1d(field, axis):
    """
    1-dimensional gradient (no inflow, just wraps)
    """
    (lst, nxt) = get_last_next(field, axis)
    return (nxt - lst) / (2 * space_granularity)


def gradient(field):
    """
    Gradient for 2D scalar field
    """
    x_grad = gradient_1d(field, 0)
    y_grad = gradient_1d(field, 1)

    return scalar_to_vec(x_grad, y_grad)


def scalar_gradient(field):
    # Use [:, :, i:i+1] to give the result the shape (_, _, 1) instead of
    # (_, _)
    x_grad = gradient_1d(field, 0)[:, :, 0:1]
    y_grad = gradient_1d(field, 1)[:, :, 1:2]

    return x_grad + y_grad


def laplacian_1d(field, axis):
    """
    One component of the laplacian for a uniform-size scalar field
    """
    (lst, nxt) = get_last_next(field, axis)

    return lst + nxt - field * 2


def laplacian(field):
    """
    The laplacian for a uniform-size scalar field
    """
    return (
        (laplacian_1d(field, 0) + laplacian_1d(field, 1)) /
        (space_granularity ** 2)
    )


def advect_1d(vel_field, field, axis):
    lst = np.roll(field, 1, axis)
    nxt = np.roll(field, -1, axis)

    diff = nxt - lst

    vel_component = vel_field[:, :, axis]

    return (
        (space_granularity / time_granularity) *
        np.multiply(
            scalar_to_vec(vel_component),
            diff
        )
    )


def advect(vel_field, last_field, field):
    """
    2-dimensional advection given a velocity field and the current and last
    values of a quantity field
    NOTE: Uses eulerian explicit method, which is unstable at low
          granularities (but means that we don't have to do sampling, which
          works great on the GPU but not so much in numpy).
    """
    advect_y = advect_1d(vel_field, field, 0)
    advect_x = advect_1d(vel_field, field, 1)

    return last_field - advect_y - advect_x


def diffuse(vel_field):
    """
    2-dimensional diffusion given a velocity field
    NOTE: Uses eulerian explicit method, which is unstable at low
          granularities.
    """
    return vel_field + viscosity * time_granularity * laplacian(vel_field)


def iterate(iter_func, initial):
    acceptable_error = 1e-5
    max_iterations = 50
    last = initial

    for _ in range(max_iterations):
        cur = iter_func(last)
        error = weighted_rms(cur, last)

        if error < acceptable_error:
            return cur

        last = cur

    return last


def mk_iterate_pressure(velocity):
    vel_grad   = scalar_gradient(velocity)
    vel_weight = -(space_granularity ** 2)
    divisor    = 4

    def output(pressure):
        lst_x = np.roll(pressure, 1,  0)
        nxt_x = np.roll(pressure, -1, 0)
        lst_y = np.roll(pressure, 1,  1)
        nxt_y = np.roll(pressure, -1, 1)

        return (
            (lst_x + nxt_x + lst_y + nxt_y + vel_weight * vel_grad) /
            divisor
        )

    return output


def apply_boundaries(scale, mat):
    # Keep It Immutable, Stupid
    new_mat = np.copy(mat)
    new_mat[:,  0] = scale * mat[:,  1]
    new_mat[:, -1] = scale * mat[:, -2]
    new_mat[0,  :] = scale * mat[1,  :]
    new_mat[-1, :] = scale * mat[-2, :]

    new_mat[0,   0] = scale * mat[1,   1]
    new_mat[-1, -1] = scale * mat[-2, -2]
    new_mat[0,  -1] = scale * mat[1,  -2]
    new_mat[-1,  0] = scale * mat[-2,  1]

    return new_mat

# Velocity grid
vel_grid = np.ones(vel_shape)
vel_grid[:, :, 0] = 0

# Gravity grid for a single time step
gra_grid = np.ones(vel_shape) * time_granularity
gra_grid[:, :, 1] = 0

# Pressure grid
pre_grid = np.zeros(pre_shape)

last_vel = vel_grid

for _ in range(1000):
    def print_speed(vel):
        vx    = vel[:, :, 0]
        vy    = vel[:, :, 1]
        speed = vx * vx + vy * vy

        print(np.sum(speed))

    tmp_grid = advect(
        vel_grid,
        last_vel,
        vel_grid
    )
    print_speed(tmp_grid)

    tmp_grid = diffuse(tmp_grid)
    print_speed(tmp_grid)

    tmp_prsr = iterate(
        mk_iterate_pressure(tmp_grid),
        pre_grid
    )

    pre_grad = gradient(tmp_prsr)

    tmp_grid -= pre_grad

    last_vel = vel_grid
    vel_grid = apply_boundaries(-1., tmp_grid)

    pre_grid = apply_boundaries(1., tmp_prsr)

    cv2.imshow("Show", np.concatenate((pre_grid, vel_grid), axis=2))
    cv2.waitKey(1)
