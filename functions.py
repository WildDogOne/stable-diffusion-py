import re

import requests
from PIL import Image

from functions import *


def display_image_in_notebook(image, scale):
    display(image.resize((int(image.width * scale), int(image.height * scale))))


def generate_image(url, payload, outfile):
    """
    This function generates images and displays them inline in the notebook
    """
    import io
    import base64
    from PIL import Image, PngImagePlugin
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        # from IPython import display
        scale = 0.5
        # scale = 1
        display(image.resize((int(image.width * scale), int(image.height * scale))))


def output_file_to_disk(subfolder, save_file_folder, image, pnginfo, outfile):
    import os
    folder = "data/" + save_file_folder + "/" + str(subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    image.save(folder + '/' + str(outfile) + '.png', pnginfo=pnginfo)


def generate_image_new(url, payload=None, outfile=None, subfolder=None, display_image=True, save_file=False,
                       save_file_folder=None, display_image_scale=0.5):
    import io
    import base64
    from PIL import Image, PngImagePlugin

    import os
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    for i in r['images']:

        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        if save_file:
            output_file_to_disk(subfolder, save_file_folder, image, pnginfo, outfile)
        if display_image:
            display_image_in_notebook(image, display_image_scale)


def set_model(url, model=None):
    """
    This function sets the model to something provided
    If nothing is provided, will just return true
    """
    if model:
        print(f"Change to model {model}")
        option_payload = {
            "sd_model_checkpoint": model
        }
        response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
        if response.status_code == 200:
            return True
        else:
            print("Failed to set model")
            return False
    else:
        print("No Model Change necessary")
        return True


def generate_images(url, samplers, payload, model=None, artists=None, keyword_test_keyword=None, subfolder=None,
                    display_image=True, save_file=False, save_file_folder=None, display_image_scale=0.5):
    """
    This function deals with the iterations of images
    Model: None = Model will not be changed
    artists: None = No need to check artists
    keyword_test = None: Will not test for tags, set to something if you want to test in combination with tags
    """
    regexpattern = r"(.*)?_(\d+)?_training_images_(\d+)?_"
    prompt = payload["prompt"]

    x = re.search(regexpattern, model)
    if x:
        x = x.groups()
        if len(x) == 3:
            model_name, pictures, training_steps = x
    if "model_name" not in locals():
        model_name = model
        pictures = "n/a"
        training_steps = "n/a"
    if set_model(url, model):
        for sampler in samplers:
            if sampler == "LMS Karras" and steps >= 90:
                print("Skipping LMS Karras, only noise from Step 90+")
            else:
                if artists:
                    for artist in artists:
                        payload["prompt"] = prompt + ", by " + artist
                        # generate_image(url, payload, f"{payload['seed']}_{sampler}")
                        generate_image_new(url, payload, display_image_scale=display_image_scale, save_file=save_file,
                                           subfolder=subfolder, display_image=display_image,
                                           save_file_folder=save_file_folder,
                                           outfile=f"artist-{artist}_model-{model}_sampler-{sampler}_steps-{payload['steps']}_seed-{payload['seed']}")
                        print(
                            f"Artist: '{artist}' Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
                elif keyword_test_keyword:
                    keywords = payload["prompt"].split(", ")
                    for keyword in keywords:
                        payload["prompt"] = keyword_test_keyword + ", " + keyword
                        # generate_image(url, payload, f"{payload['seed']}_{sampler}")
                        generate_image_new(url, payload, display_image_scale=display_image_scale, save_file=save_file,
                                           subfolder=subfolder, display_image=display_image,
                                           save_file_folder=save_file_folder,
                                           outfile=f"keyword-{keyword}_model-{model}_sampler-{sampler}_steps-{payload['steps']}_seed-{payload['seed']}")
                        print(
                            f"Keyword: '{keyword}' Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
                else:
                    payload["sampler_name"] = sampler
                    # generate_image(url, payload, f"{payload['seed']}_{sampler}")
                    generate_image_new(url, payload, display_image_scale=display_image_scale, save_file=save_file,
                                       subfolder=subfolder, display_image=display_image,
                                       save_file_folder=save_file_folder,
                                       outfile=f"model-{model}_sampler-{sampler}_steps-{payload['steps']}_seed-{payload['seed']}")
                    print(
                        f"Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
        print("done")


def initialise(seed, prompt, negative_prompt, steps, samplers=None):
    """
    Function generates a default payload, and sets default sampler list if none is provided
    """
    if seed == -1:
        import random
        seed = random.randint(1000000000, 10000000000)
        # seed = random.randint(1000000000,10000000000)
    payload = {
        "enable_hr": False,
        # "denoising_strength": 0,
        # "firstphase_width": 0,
        # "firstphase_height": 0,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        # "styles": [
        #  "Good Baseline NSFW",
        #  "Good Baseline NSFW"
        # ],
        "seed": seed,
        # "subseed": -1,
        # "subseed_strength": 0,
        # "seed_resize_from_h": -1,
        # "seed_resize_from_w": -1,
        # "sampler_name": "string",
        "batch_size": 1,
        "n_iter": 1,
        "steps": steps,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": False,
        "tiling": False,
        # "eta": 0,
        # "s_churn": 0,
        # "s_tmax": 0,
        # "s_tmin": 0,
        # "s_noise": 1,
        # "override_settings": {},
        "sampler_index": "Euler a"
    }

    if not samplers:
        samplers = [
            "Euler a",
            "Euler",
            "LMS",
            "Heun",
            "LMS Karras",
            "DPM fast",
            "DPM adaptive",
            "DPM2",
            "DPM2 a",
            "DPM2 Karras",
            "DPM2 a Karras",
            "DPM++ 2S a Karras",
            "DPM++ 2M Karras",
            "DPM++ 2S a",
            "DPM++ 2M",
            "DDIM",
            "PLMS"
        ]
    return payload, samplers
