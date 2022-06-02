import os

import matplotlib.pyplot as plt

from neural_video.data_processing.load_data import load_training_data
from neural_video.model.basic import BasicVideoRenderer

x, y = load_training_data(os.path.join("data", "frames"))

settings = {"epochs": 5001, "batch_size": 2, "learning_rate": 0.001}

model = BasicVideoRenderer(settings=settings)
history = model.train(x, y)
model.save_model()

plt.plot(history["loss_train"])
plt.show()
