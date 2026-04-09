from numpy import loadtxt
from keras.models import model_from_json

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Load model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)


model.load_weights("model.weights.h5")

print("Loaded model from disk")

# Predict probabilities
predictions = model.predict(x)


predicted_labels = (predictions > 0.5).astype(int)

# Print results
for i in range(5, 10):
    print(f"{x[i].tolist()} => {predicted_labels[i][0]} (expected {int(y[i])})")
