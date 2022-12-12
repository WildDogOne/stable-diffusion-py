def generate_image(payload, outfile):
    """
    This function generates images and displays them inline in the notebook
    """
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        from pprint import pprint
        #from IPython import display
        scale = 0.5
        scale = 1
        display(image.resize((int(image.width * scale), int(image.height * scale))))


def set_model(model=None):
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


def generate_images(samplers, payload, model=None, artists=None, keyword_test=None):
    """
    This function deals with the iterations of images
    Model: None = Model will not be changed
    artists: None = No need to check artists
    keyword_test = None: Will not test for tags, set to something if you want to test in combination with tags
    """
    regexpattern = r"(.*)?_(\d+)?_training_images_(\d+)?_"
    prompt = payload["prompt"]
    try:
        model_name, pictures, training_steps = re.search(regexpattern, model).groups()
    except:
        model_name = model
        pictures = "n/a"
        training_steps = "n/a"
    if set_model(model):
        for sampler in samplers:
            if sampler == "LMS Karras" and steps >= 90:
                print("Skipping LMS Karras, only noise from Step 90+")
            else:
                if artists:
                    for artist in artists:
                        payload["prompt"] = prompt + ", by " + artist
                        generate_image(payload, f"{payload['seed']}_{sampler}")
                        print(
                            f"Artist: '{artist}' Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
                elif keyword_test:
                    keywords = payload["prompt"].split(", ")
                    for keyword in keywords:
                        payload["prompt"] = keyword_test + ", " + keyword
                        generate_image(payload, f"{payload['seed']}_{sampler}")
                        print(
                            f"Keyword: '{keyword}' Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
                else:
                    payload["sampler_name"] = sampler
                    generate_image(payload, f"{payload['seed']}_{sampler}")
                    print(
                        f"Model Name: '{model_name}' Picture Count: {pictures} Training Steps: {training_steps}\nSampler: '{sampler}' Steps: {payload['steps']} Seed: {payload['seed']}")
        print("done")