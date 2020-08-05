import json

from models.prediction import predict

with open('sample/sample2_0.json') as f:
    data = json.load(f)

predict(data)