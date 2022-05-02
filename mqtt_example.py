"""MQTT Tx/Rx tranmission example"""
import paho.mqtt.client as mqtt
import time
import sys

import paho.mqtt.publish as publish

import connection

from threading import Thread

dist_idx = (0, 4)
gps_idx = (773, 774)
mock_t = [float(i) for i in range(775)]
mock_arr = str(mock_t)
DATA_BUFFER = connection.Buffer(dist_idx, gps_idx)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() - if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("HeatSeekingCar/rx_thermal")
    client.subscribe("HeatSeekingCar/rx_distances")
    client.subscribe("HeatSeekingCar/rx")
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

    if msg.topic == "HeatSeekingCar/rx_thermal":
        DATA_BUFFER.thermal = msg.payload

    elif msg.topic == "HeatSeekingCar/rx_distances":
        DATA_BUFFER.distance = msg.payload

    elif msg.topic == "HeatSeekingCar/rx":
        DATA_BUFFER.msg = msg.payload
        DATA_BUFFER.msg_flag = True

# mock_lat = 11.99999999999
# mock_long = 10.99999999999
# path = r"/Users/amelyaavdeyev/capstone_data/877.jpg"

# f=open(path, "rb")
# fileContent = f.read()
# byteArr = bytearray(fileContent)

# mock_msg = [0, 1, 2, 3, mock_lat, mock_long]
# print(f"SENT: {mock_msg}")

# publish.single("HeatSeekingCar/tx", byteArr, hostname="test.mosquitto.org")

# publish.single("HeatSeekingCar/tx", f"{mock_arr}", hostname="test.mosquitto.org")
# print("Done")

# Create an MQTT client and attach our routines to it.
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("test.mosquitto.org", 1883, 60)

client.loop_start()

# init env

test_turns = iter([0, 1, 2, 3, 0, 1, 2, 3, 4])
file_path = 'observation_data.txt'
sys.stdout = open(file_path, "a")


# Timeout after 15 mins
t_end = time.time() + 60 * 15
while time.time() < t_end:
    time.sleep(2.5)
    print(DATA_BUFFER.msg_flag)
    if DATA_BUFFER.msg_flag: # If buffer not empty, send it to data pipeline
        print(f"Message Recieved\nConverting Buffer to Arrays...")
        # Translate from raw data to buffer
        tick = time.time()
        t_array, d_array, lat, long = DATA_BUFFER.buffer_to_arrays()
        tock = time.time()
        print(f"\n>Process Time: {tock - tick}\n> {d_array}\n({lat}, {long})")

        # Pass Info to Data Pipeline, return the pipeline object
        # get_next_turn_thread = Thread(target=env.step(prev_turn))

        # Pass pipeline object to environment and take a step
        # environment.step

        # Recieve Next Turn
        mock_turn = next(test_turns)

        # Transmit Next Turn
        print("Transmitting Next Turn to RC car...")
        publish.single("HeatSeekingCar/tx", f"{mock_turn}", hostname="test.mosquitto.org")
        DATA_BUFFER.msg_flag = False

# A second stop loop will be if the target reached, or num_turns is at max
client.loop_stop()