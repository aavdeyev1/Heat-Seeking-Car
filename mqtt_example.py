"""Script to generate next turn for heat-seeking autonomous car."""

from ast import Pass
import paho.mqtt.client as mqtt

import paho.mqtt.publish as publish

import connection

DATA = connection.Buffer()

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() - if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("HeatSeekingCar/rx_thermal")
    client.subscribe("HeatSeekingCar/rx_distances")
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

    if msg.topic == "HeatSeekingCar/rx_thermal":
        DATA.thermal = msg.payload

    elif msg.topic == "HeatSeekingCar/rx_distances":
        DATA.distance = msg.payload


mock_distance = [i for i in range(4)]
mock_lat = 11.99999999999
mock_long = 10.99999999999
path = r"/Users/amelyaavdeyev/capstone_data/877.jpg"

f=open(path, "rb")
fileContent = f.read()
byteArr = bytearray(fileContent)

mock_msg = [0, 1, 2, 3, mock_lat, mock_long]
print(f"SENT: {mock_msg}")

# publish.single("HeatSeekingCar/test", byteArr, hostname="test.mosquitto.org")
# publish.single("HeatSeekingCar/test", f"{mock_msg}", hostname="test.mosquitto.org")
# print("Done")

# Recieve Message
car = mqtt.Client()

# Create an MQTT client and attach our routines to it.
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

 
client.connect("test.mosquitto.org", 1883, 60)

client.loop_start()

import time
t_end = time.time() + 60 * 15
while time.time() < t_end:
    time.sleep(1)
    print(DATA)


# Create Pipeline object, feed message from client
client.loop_stop()