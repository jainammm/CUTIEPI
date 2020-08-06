import json

from models.prediction import predict

with open('sample/Sample24_0.json') as f:
    data = json.load(f)

predict(data)