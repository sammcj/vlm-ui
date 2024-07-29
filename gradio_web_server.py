import argparse
import datetime
import json
import os
import time
import hashlib
import re
import threading
import random
from filelock import FileLock
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import gradio as gr
import requests

from constants import (
    CONTROLLER_URL,
    WORKER_HOST,
    GRADIO_HOST,
    GRADIO_PORT,
    LOG_LEVEL
)
from utils import (
    build_logger,
    server_error_msg,
    load_image_from_base64,
    get_log_filename,
)
from conversation import Conversation

logger = build_logger("gradio_web_server", "gradio_web_server.log")
headers = {"User-Agent": "VLM Client"}
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

logger.setLevel(LOG_LEVEL)

def write2file(path, content):
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "a") as fout:
            fout.write(content)


def sort_models(models):
    def custom_sort_key(model_name):
        if model_name == "InternVL-Chat-V1-5":
            return (1, model_name)
        elif model_name.startswith("InternVL-Chat-V1-5-"):
            return (1, model_name)
        else:
            return (0, model_name)

    models.sort(key=custom_sort_key, reverse=True)
    try:
        first_three = models[:4]
        random.shuffle(first_three)
        models[:4] = first_three
    except:
        pass
    return models

def fetch_worker_status(worker_name, timeout=5):
    try:
        r = requests.post(worker_name + '/worker_get_status', timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        logger.error(f'Failed to fetch status for worker: {worker_name}, error: {e}')
    except ValueError as e:
        logger.error(f'Failed to parse worker status JSON: {worker_name}, error: {e}')
    return None

def get_model_list():
    logger.info(f"Call `get_model_list`")
    logger.info(f"Fetching model list from controller at {args.controller_url}")
    for attempt in range(10):
        ret = requests.post(args.controller_url + "/refresh_all_workers")
        logger.info(f"status_code from `get_model_list`: {ret.status_code}")
        logger.info(f"Refresh workers response: {ret.text}")
        if ret.status_code != 200:
            logger.warning(f"Failed to refresh workers, status code: {ret.status_code}")
            time.sleep(3)
            continue

        ret = requests.post(args.controller_url + "/list_models")
        logger.info(f"status_code from `list_models`: {ret.status_code}")
        logger.info(f"List models response: {ret.text}")
        if ret.status_code != 200:
            logger.warning(f"Failed to list models, status code: {ret.status_code}")
            time.sleep(3)
            continue

        models = ret.json()["models"]
        logger.info(f"Received models: {models}")
        if not models:
            logger.warning("Received empty model list, retrying...")
            time.sleep(3)
            continue

        models = sort_models(models)
        logger.info(f"Models (from {args.controller_url}): {models}")
        return models

    logger.error("Failed to get model list after multiple attempts")
    return ["No models available"]


def init_state(state=None):
    if state is not None:
        del state
    return Conversation()


def find_bounding_boxes(state, response):
    pattern = re.compile(r"<ref>\s*(.*?)\s*</ref>\s*<box>\s*(\[\[.*?\]\])\s*</box>")
    matches = pattern.findall(response)
    results = []
    for match in matches:
        results.append((match[0], eval(match[1])))
    returned_image = None
    latest_image = state.get_images(source=state.USER)[-1]
    returned_image = latest_image.copy()
    width, height = returned_image.size
    draw = ImageDraw.Draw(returned_image)
    font = ImageFont.truetype("assets/BMNF-Regular.ttf", int(20 * line_width / 2))
    for result in results:
        line_width = max(1, int(min(width, height) / 200))
        random_color = (
            random.randint(0, 128),
            random.randint(0, 128),
            random.randint(0, 128),
        )
        category_name, coordinates = result
        coordinates = [
            (
                float(x[0]) / 1000,
                float(x[1]) / 1000,
                float(x[2]) / 1000,
                float(x[3]) / 1000,
            )
            for x in coordinates
        ]
        coordinates = [
            (
                int(x[0] * width),
                int(x[1] * height),
                int(x[2] * width),
                int(x[3] * height),
            )
            for x in coordinates
        ]
        for box in coordinates:
            draw.rectangle(box, outline=random_color, width=line_width)
            text_size = font.getbbox(category_name)
            text_width, text_height = (
                text_size[2] - text_size[0],
                text_size[3] - text_size[1],
            )
            text_position = (box[0], max(0, box[1] - text_height))
            draw.rectangle(
                [
                    text_position,
                    (text_position[0] + text_width, text_position[1] + text_height),
                ],
                fill=random_color,
            )
            draw.text(text_position, category_name, fill="white", font=font)
    return returned_image if len(matches) > 0 else None


def query_image_generation(response, sd_worker_url, timeout=15):
    if not sd_worker_url:
        return None
    sd_worker_url = f"{sd_worker_url}/generate_image/"
    pattern = r"```drawing-instruction\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        payload = {"caption": match.group(1)}
        print("drawing-instruction:", payload)
        response = requests.post(sd_worker_url, json=payload, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    else:
        return None


def load_demo(url_params, request: gr.Request = None):
    if request:
        logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    models = get_model_list()
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = init_state()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request = None):
    if request:
        logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = init_state()
    dropdown_update = gr.Dropdown(
        choices=models, value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, liked, model_selector, request: gr.Request):
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": liked,
        "model": model_selector,
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, True, model_selector, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, False, model_selector, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3

def stop_generation(state):
    state.skip_next = True
    return (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=True),
    ) + (enable_btn,) * 5 + (disable_btn,)

def vote_selected_response(
    state, model_selector, request: gr.Request, data: gr.LikeData
):
    logger.info(
        f"Vote: {data.liked}, index: {data.index}, value: {data.value} , ip: {request.client.host}"
    )
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": data.liked,
        "index": data.index,
        "model": model_selector,
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")
    return


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.update_message(Conversation.ASSISTANT, None, -1)
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = init_state()
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def change_system_prompt(state, system_prompt, request: gr.Request):
    logger.info(f"Change system prompt. ip: {request.client.host}")
    state.set_system_message(system_prompt)
    return state


def add_text(state, message, system_prompt, model_selector, request: gr.Request):
    print(f"state: {state}")
    if not state:
        state = init_state()
    images = message.get("files", [])
    text = message.get("text", "").strip()
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    textbox = gr.MultimodalTextbox(value=None, interactive=False)
    if len(text) <= 0 and len(images) == 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 6
    images = [Image.open(path).convert("RGB") for path in images]

    if len(images) > 0 and len(state.get_images(source=state.USER)) > 0:
        state = init_state(state)
    state.set_system_message(system_prompt)
    state.append_message(Conversation.USER, text, images)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), textbox, model_selector) + (disable_btn,) * 6



def http_bot(
    state,
    model_selector,
    temperature,
    top_p,
    repetition_penalty,
    max_new_tokens,
    max_input_tiles,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    if hasattr(state, "skip_next") and state.skip_next:
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
        ) + (disable_btn,) * 5 + (enable_btn,)
        return

    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    if worker_addr == "":
        state.update_message(Conversation.ASSISTANT, server_error_msg)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
        ) + (disable_btn,) * 6
        return

    all_images = state.get_images(source=state.USER)
    all_image_paths = [state.save_image(image) for image in all_images]

    pload = {
        "model": model_name,
        "prompt": state.get_prompt(),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": max_new_tokens,
        "max_input_tiles": max_input_tiles,
        "repetition_penalty": repetition_penalty,
        "images": f"List of {len(all_images)} images: {all_image_paths}",
    }
    logger.info(f"==== request ====\n{pload}")
    pload.pop("images")
    pload["prompt"] = state.get_prompt(inlude_image=True)
    state.append_message(Conversation.ASSISTANT, state.streaming_placeholder)
    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=False),
    ) + (disable_btn,) * 6

    try:
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=20,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    if "text" in data:
                        output = data["text"].strip()
                        output += state.streaming_placeholder

                    image = None
                    if "image" in data:
                        image = load_image_from_base64(data["image"])
                        _ = state.save_image(image)

                    state.update_message(Conversation.ASSISTANT, output, image)
                    yield (
                        state,
                        state.to_gradio_chatbot(),
                        gr.MultimodalTextbox(interactive=False),
                    ) + (disable_btn,) * 5 + (enable_btn,)  # Enable stop button
                else:
                    output = (
                        f"**{data['text']}**" + f" (error_code: {data['error_code']})"
                    )

                    state.update_message(Conversation.ASSISTANT, output, None)
                    yield (
                        state,
                        state.to_gradio_chatbot(),
                        gr.MultimodalTextbox(interactive=True),
                    ) + (enable_btn,) * 5 + (disable_btn,)  # Disable stop button
                    return
    except requests.exceptions.RequestException as e:
        state.update_message(Conversation.ASSISTANT, server_error_msg, None)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=True),
        ) + (enable_btn,) * 5 + (disable_btn,)  # Disable stop button
        return


    ai_response = state.return_last_message()
    if "<ref>" in ai_response:
        returned_image = find_bounding_boxes(state, ai_response)
        returned_image = [returned_image] if returned_image else []
        state.update_message(Conversation.ASSISTANT, ai_response, returned_image)
    if "```drawing-instruction" in ai_response:
        returned_image = query_image_generation(
            ai_response, sd_worker_url=sd_worker_url
        )
        returned_image = [returned_image] if returned_image else []
        state.update_message(Conversation.ASSISTANT, ai_response, returned_image)

    state.end_of_current_turn()

    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=True),
    ) + (enable_btn,) * 5 + (disable_btn,)  # Disable stop button at the end

    finish_tstamp = time.time()
    logger.info(f"{output}")
    data = {
        "tstamp": round(finish_tstamp, 4),
        "like": None,
        "model": model_name,
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "state": state.dict(),
        "images": all_image_paths,
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(data) + "\n")



title_html = """
<h2> <span class="gradient-text" id="text">VLM UI</span></h2>
<a href="https://smcleod.net">[🧑‍💻 smcleod.net]</a>
"""

learn_more_markdown = """
### Acknowledgement
This web app borrows from both LLaVA and InternVLM demos. Thanks for their awesome work!
"""

block_css = """
.gradio-container {margin: 0.1% 1% 0 1% !important; max-width: 98% !important;};
#buttons button {
    min-width: min(120px,100%);
}

.gradient-text {
    font-size: 26px;
    width: auto;
    font-weight: bold;
    background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}

.plain-text {
    font-size: 20px;
    width: auto;
    font-weight: bold;
    font-family: 'helvetica neue';
}

"""

class VLMInterface:
    def __init__(self):
        self.model_selector = None

    def periodic_refresh_models(self):
        while True:
            if self.model_selector is None or self.model_selector.choices == ["No models available"]:
                time.sleep(10)
                models = get_model_list()
                if models and models != ["No models available"] and self.model_selector is not None:
                    self.model_selector.choices = models
                    self.model_selector.value = models[0]

    def build_demo(self, embed_mode):
        textbox = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image", "video"],
            placeholder="Enter message or upload file...",
            show_label=False,
        )

        with gr.Blocks(
            title="VLM UI",
            # theme=gr.themes.Default(),
            # theme='gstaff/xkcd',
            theme='bethecloud/storj_theme',
            css=block_css,
        ) as gradio_app:
            models = get_model_list()
            state = gr.State(init_state())

            if not embed_mode:
                gr.HTML(title_html)

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id="model_selector_row"):
                        model_selector = gr.Dropdown(
                            choices=models if models else ["No models available"],
                            value=models[0] if models else "No models available",
                            interactive=True,
                            show_label=False,
                            container=False,
                        )

                    with gr.Accordion("System Prompt", open=True) as system_prompt_row:
                        system_prompt = gr.Textbox(
                            value=os.getenv(
                                "SYSTEM_MESSAGE",
                                "You are a multimodal large language model with the ability to understand images. Answer questions concisely.",
                            ),
                            label="System Prompt",
                            lines=5,
                            container=False,
                            interactive=True,
                        )
                    with gr.Accordion("Parameters", open=True) as parameter_row:
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            interactive=True,
                            label="Top P",
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=1.5,
                            value=1.1,
                            step=0.02,
                            interactive=True,
                            label="Repetition penalty",
                        )
                        max_output_tokens = gr.Slider(
                            minimum=128,
                            maximum=8192,
                            value=2048,
                            step=64,
                            interactive=True,
                            label="Max output tokens",
                        )
                        max_input_tiles = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=12,
                            step=1,
                            interactive=True,
                            label="Max input tiles (control the image size)",
                        )

                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="InternVL2",
                        height=920,
                        show_copy_button=True,
                        show_share_button=True,
                        avatar_images=[
                            "assets/human.png",
                            "assets/assistant.png",
                        ],
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        with gr.Column(scale=8):
                            textbox.render()
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Send", variant="primary")

                    with gr.Row(elem_id="buttons") as button_row:
                        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                        stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                        clear_btn = gr.Button(value="🗑️  Clear", interactive=False)


            if not embed_mode:
                gr.Markdown(learn_more_markdown)
            url_params = gr.JSON(visible=False)

            # Register listeners
            btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn, stop_btn]

            upvote_btn.click(
                upvote_last_response,
                [state, model_selector],
                [textbox, upvote_btn, downvote_btn],
            )
            downvote_btn.click(
                downvote_last_response,
                [state, model_selector],
                [textbox, upvote_btn, downvote_btn],
            )
            chatbot.like(
                vote_selected_response,
                [state, model_selector],
                [],
            )

            # Define the http_bot event
            http_bot_event = textbox.submit(
                add_text,
                [state, textbox, system_prompt, model_selector],
                [state, chatbot, textbox, model_selector] + btn_list,
            ).then(
                http_bot,
                [
                    state,
                    model_selector,
                    temperature,
                    top_p,
                    repetition_penalty,
                    max_output_tokens,
                    max_input_tiles,
                ],
                [state, chatbot, textbox] + btn_list,
            )

            regenerate_btn.click(
                regenerate,
                [state, system_prompt],
                [state, chatbot, textbox] + btn_list,
            ).then(
                http_bot,
                [
                    state,
                    model_selector,
                    temperature,
                    top_p,
                    repetition_penalty,
                    max_output_tokens,
                    max_input_tiles,
                ],
                [state, chatbot, textbox] + btn_list,
            )
            clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

            submit_btn.click(
                add_text,
                [state, textbox, system_prompt, model_selector],
                [state, chatbot, textbox, model_selector] + btn_list,
            ).then(
                http_bot,
                [
                    state,
                    model_selector,
                    temperature,
                    top_p,
                    repetition_penalty,
                    max_output_tokens,
                    max_input_tiles,
                ],
                [state, chatbot, textbox] + btn_list,
            )
            stop_btn.click(
                # FIXME: this just causes the webui to stop, it doesn't tell the worker to stop - I need to fix that
                stop_generation,
                inputs=[state],
                outputs=[state, chatbot, textbox] + btn_list,
                cancels=[http_bot_event]  # This cancels the running http_bot event
            )

        # Start the periodic refresh thread
        threading.Thread(target=self.periodic_refresh_models, daemon=True).start()
        return gradio_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=GRADIO_HOST)
    parser.add_argument("--port", type=int, default=GRADIO_PORT)
    parser.add_argument(
        "--controller-url",
        type=str,
        default=CONTROLLER_URL,
    )
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="reload", choices=["once", "reload"]
    )
    parser.add_argument("--sd-worker-url", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--worker-ip", type=str, default=WORKER_HOST)

    args = parser.parse_args()
    logger.info(f"args: {args}")
    if not args.controller_url:
        args.controller_url = os.environ.get("CONTROLLER_URL", 'http://0.0.0.0:21001')

    if not args.controller_url:
        raise ValueError("controller-url is required.")

    if not args.worker_ip:
        args.worker_ip = os.environ.get("WORKER_HOST", WORKER_HOST)

    sd_worker_url = args.sd_worker_url
    logger.info(args)

    vlm_interface = VLMInterface()
    app = vlm_interface.build_demo(args.embed)
    app.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=args.concurrency_count,
    )
