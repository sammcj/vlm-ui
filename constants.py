import os


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = 'logs/'

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<IMG_CONTEXT>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
IMAGE_PLACEHOLDER = '<image-placeholder>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

server_error_msg = "**ERROR. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

### Controller Constants
CONTROLLER_URL = "http://0.0.0.0:21001" # Do use port here
CONTROLLER_HOST = "0.0.0.0"
CONTROLLER_PORT = "21001" #10075

### Worker Constants
WORKER_URL = "http://0.0.0.0" # Don't use port here
WORKER_HOST = "0.0.0.0"
WORKER_PORT = 21002

### Gradio Constants
GRADIO_HOST = "0.0.0.0" # Don't use port here
GRADIO_PORT = 7866

### Model Constants
MODEL_NAME = os.getenv("MODEL_NAME")
# MODEL_PATH = os.getenv("MODEL_PATH")
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BITS", "False").lower() == "true"
