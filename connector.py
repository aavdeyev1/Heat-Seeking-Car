""""Object to Connect to the RC car and recieve and transmit data."""
import paho.mqtt.client as mqtt
import time
import tensorflow as tf
import numpy as np
import paho.mqtt.publish as publish
import buffer
from errors import SaveAndExitError

timeout_mins = 1
dist_idx = (0, 4)
gps_idx = (772, 773)
mock_t = [float(i) for i in range(774)]
mock_arr = str(mock_t)
DATA_BUFFER = buffer.Buffer(dist_idx, gps_idx)
host_name = "test.mosquitto.org"
request_path = "HeatSeekingCar/Request"
observation_data_path = "HeatSeekingCar/Observation_Vector"
turn_direction_path = "HeatSeekingCar/Direction"

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() - if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(observation_data_path)
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

    if msg.topic == observation_data_path:
        DATA_BUFFER.msg = msg.payload
        DATA_BUFFER.msg_flag = True


class Connector():

    def __init__(self):
        self.data_buffer = DATA_BUFFER

        # Create an MQTT client and attach our routines to it.
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_message = on_message

        self.client.connect(host_name, 1883, 60)

        self.client.loop_start()

        # Timeout after 15 mins
        self.t_end = time.time() + 60 * timeout_mins

        print("Sending connection request...")
        publish.single(request_path, "1", hostname=host_name)

    def check_for_data(self):
        publish.single(request_path, "1", hostname=host_name)
        print(self.data_buffer.msg_flag)
        if self.data_buffer.msg_flag: # If buffer not empty, send it to data pipeline
            print(f"Message Recieved\nConverting Buffer to Arrays...")
            # Translate from raw data to buffer
            tick = time.time()
            t_array, d_array, lat, long = DATA_BUFFER.buffer_to_arrays()
            tock = time.time()
            print(f"\n>Process Time: {tock - tick}\n> {d_array}\n({lat}, {long})")
            print(self.data_buffer.msg)

            t_array, d_array = self.preprocess_data(t_array, d_array)
            
            return t_array, d_array, lat, long

        else:
            return mock_arr, mock_arr, None, None

    def preprocess_data(self, t_array, d_array):
        # Preprocessing: averaging filter on t_array, reduce size by half
        pre_t_array = tf.convert_to_tensor(
                t_array, dtype=np.float32, dtype_hint=None, name=None)
        pre_t_array = tf.reshape(pre_t_array, (1, 24, 32, 1))
        avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                strides=(2, 2), padding='valid')
        pre_t_array = avg_pool_2d(pre_t_array)
        pre_t_array = tf.reshape(pre_t_array, (12, 16))

        # Round all values of t_array
        tf.round(pre_t_array)

        for idx, element in enumerate(d_array):
            d_array[idx] = round(element/10)*10
        
        return pre_t_array, d_array
            
    def send_turn(self, action: str):
        # If first attempt to send turn cause error, try again. else raise SaveAndExitError.
        # Transmit Next Turn
        print("Transmitting Next Turn to RC car...")
        try:
            publish.single(turn_direction_path, int(action), hostname=host_name)
        except TypeError:
            try:
                publish.single(turn_direction_path, int(action), hostname=host_name)
            except TypeError:
                raise SaveAndExitError("Error In Transmitting Next Turn... Saving and Exiting...")
        finally:
            self.data_buffer.msg_flag = False


if __name__ == "__main__":
    test_turns = iter([0, 1, 2, 3, 0, 1, 2, 3, 4])
    # file_path = 'observation_data.txt'
    # sys.stdout = open(file_path, "a")
    conn = Connector()
    while 1:
        t_array, d_array, lat, long = conn.check_for_data()
        if lat != -1:
            conn.send_turn(next(test_turns))