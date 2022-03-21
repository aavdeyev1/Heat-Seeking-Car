"""Functions for tx/rx with heat seeking autonomous car.

/usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
"""
import numpy as np

t_dim = (24, 32)

class Buffer():

    def __init__(self, dist_idx, gps_idx):
        self.msg = None
        self.msg_flag = False
        self.dist_idx = dist_idx
        self.gps_idx = gps_idx
        self.thermal = ""
        self.distance = ""

    def __str__(self):
        return f"Thermal: {self.msg}, Dist: {self.distance}"

    def buffer_to_arrays(self):
        temp = []
        for element in str(self.msg).replace("'", "").replace("b", ""). replace("[", "").replace("]", "").split(", "):
            temp.append(float(element))

        t_array = np.array(temp[self.dist_idx[1]:self.gps_idx[0]], dtype=np.float32).reshape(t_dim)
        d_array = np.array(temp[ self.dist_idx[0]:self.dist_idx[1] ], dtype=np.int32)
        lat = temp[self.gps_idx[0]]
        long = temp[self.gps_idx[1]]

        return t_array, d_array, lat, long

if __name__ == "__main__":
    print("Running Check...")
    dist_idx = (0, 4)
    gps_idx = (772, 773)
    mock_t = [float(i) for i in range(775)]
    mock_arr = "b'"+str(mock_t)
    check = np.array(mock_t, dtype=np.float32)
    print(check)

    buf = Buffer(dist_idx, gps_idx)
    buf.msg = mock_arr
    print(buf.buffer_to_arrays())