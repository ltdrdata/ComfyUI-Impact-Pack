import re
import random
import os
import nodes
import folder_paths

wildcard_dict = {}


def read_wildcard_dict(wildcard_path):
    global wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = os.path.splitext(rel_path)[0].replace('\\', '/')

                with open(file_path, 'r', encoding="UTF-8") as f:
                    lines = f.read().splitlines()
                    wildcard_dict[key] = lines

    return wildcard_dict


def process(text, seed=None):
    if seed is not None:
        random.seed(seed)

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    config_value = int(parts[0])
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            replacement = random.choices(options, weights=normalized_probabilities, k=1)[0]
            replacements_found = True
            return re.sub(r'^[0-9]+::', '', replacement, 1)

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        pattern = r'\[([^[\]]*?)\]'
        replaced_string = re.sub(pattern, replace_option, replaced_string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        global wildcard_dict
        pattern = r"__([\w.\-/]+)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            if match in wildcard_dict:
                replacement = random.choice(wildcard_dict[match])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


def safe_float(x):
    try:
        return float(x)
    except:
        return 1.0


def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)
    items = [match.strip(':') for match in matches]

    result = {}
    for item in items:
        item = item.split(':')

        lora = None
        a = 1.0
        b = 1.0
        if len(item) == 1:
            lora = item[0]
        elif len(item) == 2:
            lora = item[0]
            a = safe_float(item[1])
            b = a  # When only one weight is provided, use the same weight for model as well as clip - similar to Automatic1111
        elif len(item) >= 3:
            lora = item[0]
            if item[1] != '':
                a = safe_float(item[1])
            b = safe_float(item[2])

        if lora is not None:
            result[lora] = a, b

    return result


def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result


def process_with_loras(wildcard_opt, model, clip):
    pass1 = process(wildcard_opt)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    for lora_name, (model_weight, clip_weight) in loras.items():
        if (lora_name.split('.')[-1]) not in folder_paths.supported_pt_extensions:
            lora_name = lora_name+".safetensors"

        path = folder_paths.get_full_path("loras", lora_name)

        if path is not None:
            print(f"LOAD LORA: {lora_name}: {model_weight}, {clip_weight}")
            model, clip = nodes.LoraLoader().load_lora(model, clip, lora_name, model_weight, clip_weight)
        else:
            print(f"LORA NOT FOUND: {lora_name}")

    print(f"CLIP: {pass2}")
    return model, clip, nodes.CLIPTextEncode().encode(clip, pass2)[0]

