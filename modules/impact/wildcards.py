import re
import random
import os

wildcard_dict = {}


def read_wildcard_dict(wildcard_path):
    global wildcard_dict
    for root, directories, files in os.walk(wildcard_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                key = os.path.splitext(file)[0]

                with open(file_path, 'r') as f:
                    lines = f.read().splitlines()

                wildcard_dict[key] = lines

    return wildcard_dict


def process(text):
    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')
            replacement = random.choice(options)
            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        global wildcard_dict
        pattern = r"__([\w.-]+)__"
        matches = re.findall(pattern, string)

        for match in matches:
            if match in wildcard_dict:
                replacement = random.choice(wildcard_dict[match])
                string = string.replace(f"__{match}__", replacement, 1)
        
        return string

    # phase1: replace options
    phase1, is_replaced = replace_options(text)

    while is_replaced:
        phase1, is_replaced = replace_options(phase1)

    # phase2: replace wildcards
    phase2 = replace_wildcard(phase1)
    
    return phase2


