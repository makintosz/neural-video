import os

from neural_video.data_processing.load_data import load_training_data

x, y = load_training_data(os.path.join("data", "frames"))
