from __future__ import print_function

import argparse
import time
import numpy as np
from scipy.misc import imread
import json

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from models.loaddata import DataLoader
from configs.config import model_params


def run(host, port, model, signature_name):

    with open('sample/sample2_0.json') as f:
        json_data = json.load(f)

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Read an image
    data_loader = DataLoader(model_params, json_data, update_dict=False, load_dictionary=True)

    data = data_loader.fetch_validation_data()

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['data_grid'].CopyFrom(make_tensor_proto(data['grid_table'], shape=data['grid_table'].shape))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--model', help='model name', default='CUTIE', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.model, args.signature_name)