"""An object to hold the pipelines for a given episode."""

from cgitb import reset
from pipeline import Pipeline
import numpy as np

d_array = np.array([500.0, 500.0, 500.0, 500.0], dtype=np.float32)
t_array = np.zeros((24, 32))
reset_pipeline = Pipeline(t_array=t_array, d_array=d_array)

class PipelineHistorian():

    def __init__(self):
        self.history = []
        self.current_pl = reset_pipeline
        self.prev_pl = None

    def add_step(self, pipeline: Pipeline):
        self.history.append(self.prev_pl)
        self.prev_pl = self.current_pl
        self.current_pl = pipeline

    def reset(self):
        self.history = []
        self.current_pl = reset_pipeline
        self.prev_pl = None
