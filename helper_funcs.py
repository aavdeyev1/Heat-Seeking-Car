import numpy as np

# Size of Thermal Camera Arrays
T_m = 4
T_n = 4

def copy_border(arr: np.array):
    arr_temp = np.ones((T_m + 2, T_n + 2))
    for row in range(T_m):
        for col in range(T_n):
            arr_temp[row + 1, col + 1] = arr[row, col]
    arr_temp[0, 0] = arr[0, 0] # left top corner
    arr_temp[0, -1] = arr[0, -1] # right top corner
    arr_temp[-1, 0] = arr[-1, 0] # left bot corner
    arr_temp[-1, -1] = arr[-1, -1] # right bot corner
    arr_temp[0, 1:-1] = arr[0, 0:] # first row
    arr_temp[-1, 1:-1] = arr[-1, 0:] # last row
    arr_temp[1:-1, 0] = arr[0:, 0] # first col
    arr_temp[1:-1, -1] = arr[0:, -1] # last col
    # print(arr_temp)
    return arr_temp

def normalize(arr: np.array):
    max = np.max(arr)
    arr = arr / max
    return arr

def IBO(arr: np.array):
    """Intensity Brightening Operation.
        Preprocess the Thermal Image to Brighten the bright areas
        and reduce the size of the array.
        Double IBO algo from Defect detection by Heriansyah et al
    """
    arr_temp = copy_border(arr)
    for row_idx in range(T_m):
        for col_idx in range(T_n):
            # print(f"{row_idx+1}, {col_idx+1}")
            z = 1
            z0 = arr_temp[row_idx + 1, col_idx + 1] # f(i, j)
            z1 = arr_temp[row_idx + 1, col_idx] # f(i, j-1)
            z2 = arr_temp[row_idx, col_idx + 1] # f(i-1, j)
            z3 = arr_temp[row_idx, col_idx] # f(i-1, j-1)
            z4 = arr_temp[row_idx + 2, col_idx + 2] # f(i+1, j+1)
            z5 = arr_temp[row_idx, col_idx + 2] # f(i-1, j+1)
            z6 = arr_temp[row_idx + 2, col_idx] # f(i+1, j-1)
            z7 = arr_temp[row_idx + 1, col_idx + 2] # f(i, j+1)
            z8 = arr_temp[row_idx + 2, col_idx+1] # f(i+1, j)
            # print(f"{z}")
            arr[row_idx, col_idx] = z*z0*z1*z2*z3*z4*z5*z6*z7*z8
    arr = normalize(arr)
    return arr

# def generate_QTable(thermal, distance, t_range, d_range):
#     qtable = np.zeros(((t_range[1] - t_range[0])*T_m*T_n*(d_range[1] - d_range[0])*4, 5))
#     for t_val in range(t_range[0], t_range[1]):
#         for d_val in range(d_range[0], d_range[1]):
#             for t_x_idx in range(T_n):
#                 for t_y_idx in range(T_m):
#                     for d_idx in range(3):
#                         qtable[0, :] = (np.array([]))