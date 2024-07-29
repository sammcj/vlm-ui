from ast import Dict
import logging
import logging.handlers
import os
import sys
import base64
from PIL import Image
from io import BytesIO
import json
import requests
from constants import LOGDIR
import datetime

server_error_msg = "**Server Error!"

handler = None

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def resize_image(img, max_size=800, min_size=400):
    aspect_ratio = max(img.size) / min(img.size)
    new_size = (
        int(min(max_size, max(img.size))),
        int(min(max_size / aspect_ratio, min(img.size)))
    )
    return img.resize(new_size)

def create_image_html(img, alt_text):
    img = resize_image(img)
    img_b64 = pil_to_base64(img)
    return f'<img src="data:image/png;base64,{img_b64}" alt="{alt_text}" />'

def process_stream(response):
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                yield data["text"]
            else:
                yield f"{data['text']} (error_code: {data['error_code']})"

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def get_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def data_wrapper(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, Image.Image):
        buffered = BytesIO()
        data.save(buffered, format="PNG")
        return buffered.getvalue()
    elif isinstance(data, str):
        return data.encode()
    elif isinstance(data, Dict):
        return json.dumps(data).encode()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
