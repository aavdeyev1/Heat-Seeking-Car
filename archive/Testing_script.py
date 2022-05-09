import numpy as np
import helper_funcs as hf

# init test arrays, 2D right now
test_t = np.array( [[27.0, 27.0, 27.0, 27.0],
                    [27.0, 27.0, 32.0, 32.0],
                    [27.0, 27.0, 32.0, 32.0],
                    [27.0, 27.0, 27.0, 27.0]])
test_t_cpy = np.array(test_t)
testing1 = hf.IBO(arr=test_t)
testing1_cpy = np.array(testing1)
testing2 = hf.IBO(arr=testing1)
print(f"Thermal Before:\n{test_t_cpy}\nAfter1:\n{testing1_cpy}\nAfter2:\n{testing2}\n\n")

test_d = np.array( [[300, 10, 300, 300]])
t_range = (2, 2)
d_range = (1, 2)

# Form: (distance, thermal)
QTable = hf.generate_QTable(test_t_cpy, test_d, t_range, d_range)
print(test_d)
print()