import numpy as np
import matplotlib.pyplot as plt

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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')