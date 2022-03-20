"""Functions for tx/rx with heat seeking autonomous car.
"""

import paho.mqtt.client as mqtt


class Buffer():

    def __init__(self):
        self.thermal = []
        self.distance = []

    def __str__(self):
        return f"Thermal: {len(self.thermal)}, Dist: {self.distance}"