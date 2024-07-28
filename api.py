import base64
import json
from io import BytesIO
import os

import requests
from PIL import Image
from constants import (
    CONTROLLER_URL,
)

def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    models = ret.json()["models"]
    return models


def get_selected_worker_ip(controller_url, selected_model):
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": selected_model}
    )
    worker_addr = ret.json()["address"]
    return worker_addr


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

controller_url = CONTROLLER_URL
model_list = get_model_list(controller_url)
print(f"Model list: {model_list}")

selected_model = "InternVL2-1B"
worker_addr = get_selected_worker_ip(controller_url, selected_model)
print(f"model_name: {selected_model}, worker_addr: {worker_addr}")

# send_messages = [{'role': 'system', 'content': system_message}]
# send_messages.append({'role': 'user', 'content': 'question1 to image1', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer1'})
# send_messages.append({'role': 'user', 'content': 'question2 to image2', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer2'})
# send_messages.append({'role': 'user', 'content': 'question3 to image1 & 2', 'image': []})

image = Image.open("image1.jpg")
print(f"Loading image, size: {image.size}")
system_message = os.getenv("SYSTEM_MESSAGE", "")
send_messages = [{"role": "system", "content": system_message}]
send_messages.append(
    {
        "role": "user",
        "content": "describe this image in detail",
        "image": [pil_image_to_base64(image)],
    }
)

pload = {
    "model": selected_model,
    "prompt": send_messages,
    "temperature": os.getenv("TEMPERATURE", 0.3),
    "top_p": os.getenv("TOP_P", 0.9),
    "max_new_tokens": os.getenv("MAX_NEW_TOKENS", 2048),
    "max_input_tiles": os.getenv("MAX_INPUT_TILES", 12),
    "repetition_penalty": os.getenv("REPETITION_PENALTY", 1.0),
}
headers = {"User-Agent": "InternVL-Chat Client"}
response = requests.post(
    worker_addr + "/worker_generate_stream",
    headers=headers,
    json=pload,
    stream=True,
    timeout=10,
)
for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        data = json.loads(chunk.decode())
        if data["error_code"] == 0:
            output = data["text"]
        else:
            output = data["text"] + f" (error_code: {data['error_code']})"
print(output)
