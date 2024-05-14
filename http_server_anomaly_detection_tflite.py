# import os
# import argparse
# import json
# import threading
# from functools import partial
# from http.server import HTTPServer, BaseHTTPRequestHandler
# import numpy as np
# import scipy as sp
# from scipy import stats
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model

# # For Date and Time
# import datetime

# # For Telegram Bot
# import requests
# TOKEN = '7076377067:AAE98GG2-br73-A3hq0tqD1_jQQGqxycLyc'
# CHAT_ID = "1224139730"

# # Settings
# DEFAULT_PORT = 1337
# MODELS_PATH = 'D:\\ML\\ML Project\\Try\\Auto-Encoder\\Models'
# H5_MODEL_FILE = 'D:\\ML\\ML Project\\Try\\Auto-Encoder\\Models\\New_Model.h5'
# MAX_MEASUREMENTS = 200
# ANOMALY_THRESHOLD = 1.400e-03

# # Global flag
# server_ready = 0

# # Load Keras model
# model = load_model(os.path.join(MODELS_PATH, H5_MODEL_FILE))

# # Function: extract specified features (MAD) from sample
# def extract_features(sample, max_measurements=0):
#     features = []

#     if max_measurements == 0:
#         max_measurements = sample.shape[0]
#     sample = sample[:max_measurements]

#     features.append(stats.median_abs_deviation(sample))

#     return np.array(features).flatten()

# # Decode string to JSON and save measurements in a file
# def parse_samples(json_str):
#     try:
#         json_doc = json.loads(json_str)
#     except Exception as e:
#         print('ERROR: Could not parse JSON |', str(e))
#         return

#     current_time = datetime.datetime.now().strftime("%H:%M:%S")
#     current_date = datetime.date.today().strftime("%Y-%m-%d")
#     message = f"Anomaly Detected! at {current_time} on {current_date}"
#     url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"

#     sample = np.array([[float(json_doc['x'][i]),
#                         float(json_doc['y'][i]),
#                         float(json_doc['z'][i])] for i in range(len(json_doc['x']))])

#     feature_set = extract_features(sample, max_measurements=MAX_MEASUREMENTS)
#     print("MAD:", feature_set)

#     # Make prediction from model
#     in_tensor = np.float32(feature_set.reshape(1, feature_set.shape[0]))
#     pred = model.predict(in_tensor)
#     print("Prediction:", pred.flatten())

#     # Calculate MSE
#     mse = np.mean(np.power(feature_set - pred.flatten(), 2))
#     print("MSE:", mse)

#     # Compare MSE with the threshold
#     if mse > ANOMALY_THRESHOLD:
#         print("ANOMALY DETECTED! :/")
#         requests.get(url)
#     else:
#         print("Normal :D")

# # Handler class for HTTP requests
# class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def do_GET(self):
#         # Tell client if server is ready for a new sample
#         self.send_response(200)
#         self.end_headers()
#         self.wfile.write(str(server_ready).encode())

#     def do_POST(self):
#         # Read message
#         content_length = int(self.headers['Content-Length'])
#         body = self.rfile.read(content_length)

#         # Respond with 204 "no content" status code
#         self.send_response(204)
#         self.end_headers()

#         # Decode JSON and compute MSE
#         parse_samples(body.decode('ascii'))

# # Server thread
# class ServerThread(threading.Thread):

#     def __init__(self, *args, **kwargs):
#         super(ServerThread, self).__init__(*args, **kwargs)
#         self._stop_event = threading.Event()

#     def stop(self):
#         self._stop_event.set()

#     def is_stopped(self):
#         return self._stop_event.is_set()

# # Main
# # Parse arguments
# parser = argparse.ArgumentParser(description='Server that saves data from' +
#                                     'IoT sensor node.')
# parser.add_argument('-p', action='store', dest='port', type=int,
#                     default=DEFAULT_PORT, help='Port number for server')
# args = parser.parse_args()
# port = args.port

# # Print versions
# print('Numpy ' + np.__version__)
# print('SciPy ' + sp.__version__)

# # Create server
# handler = partial(SimpleHTTPRequestHandler)
# server = HTTPServer(('', port), handler)
# server_addr = server.socket.getsockname()
# print('Server running at: ' + str(server_addr[0]) + ':' +
#         str(server_addr[1]))

# # Create thread running server
# server_thread = ServerThread(name='server_daemon',
#                             target=server.serve_forever)
# server_thread.daemon = True
# server_thread.start()

# # Store samples for a given time
# server_ready = 1
# while True:
#     pass
# print('Server shutting down')
# server.shutdown()
# server_thread.stop()
# server_thread.join()


import os
import argparse
import json
import threading
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import scipy as sp
from scipy import stats
# import tensorflow as tf
# from tensorflow import keras
from keras.models import load_model

# For Date and Time
import datetime

# For Telegram Bot
import requests
TOKEN = '7076377067:AAE98GG2-br73-A3hq0tqD1_jQQGqxycLyc'
CHAT_ID = "1224139730"

# Settings
DEFAULT_PORT = 1337
MODELS_PATH = 'D:\\ML\\ML Project\\Try\\Auto-Encoder\\Models'
H5_MODEL_FILE = 'D:\\ML\\ML Project\\Try\\Auto-Encoder\\Models\\New_Model.h5'
MAX_MEASUREMENTS = 200
# ANOMALY_THRESHOLD = 1.400e-03



# Global flag
server_ready = 0

# Load Keras model
model = load_model(os.path.join(MODELS_PATH, H5_MODEL_FILE))

# Function: extract specified features (MAD) from sample
def extract_features(sample, max_measurements=0):
    features = []

    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[:max_measurements]

    features.append(stats.median_abs_deviation(sample))

    return np.array(features).flatten()

# Decode string to JSON and save measurements in a file
def parse_samples(json_str):

    NORMAL = 1.500e-03
    WEIGHT = 0.1
    WEIGHT_TO_NORMAL = 0
    
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    normal_message = f"\U00002705 Normal Operation! \U0001F600 {current_time} on {current_date}"
    ideal_message = f"\U0001F6A8 DANGER!!! Anomaly Ideal Detected! \U000026A0 {current_time} on {current_date}"
    weight_message = f"\U0001F6A8 DANGER!!! Anomaly Weight Detected! \U000026A0 {current_time} on {current_date}"
    url_normal = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={normal_message}"
    url_ideal = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={ideal_message}"
    url_weight = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={weight_message}"
    
    try:
        json_doc = json.loads(json_str)
    except Exception as e:
        print('ERROR: Could not parse JSON |', str(e))
        return

    

    sample = np.array([[float(json_doc['x'][i]),
                        float(json_doc['y'][i]),
                        float(json_doc['z'][i])] for i in range(len(json_doc['x']))])

    feature_set = extract_features(sample, max_measurements=MAX_MEASUREMENTS)
    print("MAD:", feature_set)

    # Make prediction from model
    in_tensor = np.float32(feature_set.reshape(1, feature_set.shape[0]))
    pred = model.predict(in_tensor)
    print("Prediction:", pred.flatten())

    # Calculate MSE
    mse = np.mean(np.power(feature_set - pred.flatten(), 2))
    print("MSE:", mse)

    # Compare MSE with the threshold
    if mse < NORMAL:
        WEIGHT_TO_NORMAL = 0
        print("\U00002705 Normal Operation! \U0001F600 ")
        requests.get(url_normal)
    elif mse > WEIGHT:
        WEIGHT_TO_NORMAL = 1
        print("\U0001F6A8 DANGER!!! Anomaly Weight Detected! \U000026A0 ")
        requests.get(url_weight)
    else:
        if WEIGHT_TO_NORMAL == 1:
            WEIGHT_TO_NORMAL = 0
            print("\U00002705 Normal Operation! \U0001F600 ")
            requests.get(url_normal)
        else:
            WEIGHT_TO_NORMAL = 0
            print("\U0001F6A8 DANGER!!! Anomaly Ideal Detected! \U000026A0 ")
            requests.get(url_ideal)

# Handler class for HTTP requests
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # Tell client if server is ready for a new sample
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(server_ready).encode())

    def do_POST(self):
        # Read message
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Respond with 204 "no content" status code
        self.send_response(204)
        self.end_headers()

        # Decode JSON and compute MSE
        parse_samples(body.decode('ascii'))

# Server thread
class ServerThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(ServerThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

# Main
# Parse arguments
parser = argparse.ArgumentParser(description='Server that saves data from' +
                                    'IoT sensor node.')
parser.add_argument('-p', action='store', dest='port', type=int,
                    default=DEFAULT_PORT, help='Port number for server')
args = parser.parse_args()
port = args.port

# Print versions
print('Numpy ' + np.__version__)
print('SciPy ' + sp.__version__)

# Create server
handler = partial(SimpleHTTPRequestHandler)
server = HTTPServer(('', port), handler)
server_addr = server.socket.getsockname()
print('Server running at: ' + str(server_addr[0]) + ':' +
        str(server_addr[1]))

# Create thread running server
server_thread = ServerThread(name='server_daemon',
                            target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

# Store samples for a given time
server_ready = 1
while True:
    pass
print('Server shutting down')
server.shutdown()
server_thread.stop()
server_thread.join()
