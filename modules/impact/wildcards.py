import re
import random
import os
import nodes
import folder_paths
import yaml
import numpy as np
import threading
from impact import utils

RE_WildCardQuantifier = re.compile(r"(?P<quantifier>\d+)#__(?P<keyword>[\w.\-+/*\\]+)__", re.IGNORECASE)
wildcard_lock = threading.Lock()
wildcard_dict = {}


def get_wildcard_list():
    with wildcard_lock:
        return [f"__{x}__" for x in wildcard_dict.keys()]


def get_wildcard_dict():
    global wildcard_dict
    with wildcard_lock:
        return wildcard_dict


def wildcard_normalize(x):
    return x.replace("\\", "/").lower()


def read_wildcard(k, v):
    if isinstance(v, list):
        k = wildcard_normalize(k)
        wildcard_dict[k] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            new_key = f"{k}/{k2}"
            new_key = wildcard_normalize(new_key)
            read_wildcard(new_key, v2)


def read_wildcard_dict(wildcard_path):
    global wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = os.path.splitext(rel_path)[0].replace('\\', '/').lower()

                try:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        lines = f.read().splitlines()
                        wildcard_dict[key] = lines
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        lines = f.read().splitlines()
                        wildcard_dict[key] = lines
            elif file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

                    for k, v in yaml_data.items():
                        read_wildcard(k, v)

    return wildcard_dict


def process(text, seed=None):
    if seed is not None:
        random.seed(seed)
    random_gen = np.random.default_rng(seed)

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    b = r.group(3)
                    if b is not None:
                        b = b.strip()
                        
                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random_gen.integers(low=select_range[0], high=select_range[1]+1, size=1)

            if select_count > len(options):
                random_gen.shuffle(options)
                selected_items = options
            else:
                selected_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        local_wildcard_dict = get_wildcard_dict()
        pattern = r"__([\w.\-+/*\\]+)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in local_wildcard_dict:
                replacement = random_gen.choice(local_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+','\\+')
                total_patterns = []
                found = False
                for k, v in local_wildcard_dict.items():
                    if re.match(subpattern, k) is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random_gen.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop
        
        option_quantifier = [e.groupdict() for e in RE_WildCardQuantifier.finditer(text)]
        for match in option_quantifier:
            keyword = match['keyword'].lower()
            quantifier = int(match['quantifier']) if match['quantifier'] else 1
            replacement = '__|__'.join([keyword,] * quantifier)
            wilder_keyword = keyword.replace('*', '\\*')
            RE_TEMP = re.compile(fr"(?P<quantifier>\d+)#__(?P<keyword>{wilder_keyword})__", re.IGNORECASE)
            text = RE_TEMP.sub(f"__{replacement}__", text)

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    else:
        return 1.0


def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r'LBW=[A-Za-z][A-Za-z0-9_-]*:', r'LBW=', text)

    items = [touch_lbw(match.strip(':')) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(':')

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(';'):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != '':
                            lbw = lbw_item

        if a is None:
            a = 1.0
        if b is None:
            b = a

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b))
            added.add(lora)

    return result


def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result


def resolve_lora_name(lora_name_cache, name):
    if os.path.exists(name):
        return name
    else:
        if len(lora_name_cache) == 0:
            lora_name_cache.extend(folder_paths.get_filename_list("loras"))

        for x in lora_name_cache:
            if x.endswith(name):
                return x


def process_with_loras(wildcard_opt, model, clip, clip_encoder=None, seed=None, processed=None):
    """
    process wildcard text including loras

    :param wildcard_opt: wildcard text
    :param model: model
    :param clip: clip
    :param clip_encoder: you can pass custom encoder such as adv_cliptext_encode
    :param seed: seed for populating
    :param processed: output variable - [pass1, pass2, pass3] will be saved into passed list
    :return: model, clip, conditioning
    """

    lora_name_cache = []

    pass1 = process(wildcard_opt, seed)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b in loras:
        lora_name_ext = lora_name.split('.')
        if ('.'+lora_name_ext[-1]) not in folder_paths.supported_pt_extensions:
            lora_name = lora_name+".safetensors"

        orig_lora_name = lora_name
        lora_name = resolve_lora_name(lora_name_cache, lora_name)

        if lora_name is not None:
            path = folder_paths.get_full_path("loras", lora_name)
        else:
            path = None

        if path is not None:
            print(f"LOAD LORA: {lora_name}: {model_weight}, {clip_weight}, LBW={lbw}, A={lbw_a}, B={lbw_b}")

            def default_lora():
                return nodes.LoraLoader().load_lora(model, clip, lora_name, model_weight, clip_weight)

            if lbw is not None:
                if 'LoraLoaderBlockWeight //Inspire' not in nodes.NODE_CLASS_MAPPINGS:
                    utils.try_install_custom_node(
                        'https://github.com/ltdrdata/ComfyUI-Inspire-Pack',
                        "To use 'LBW=' syntax in wildcards, 'Inspire Pack' extension is required.")

                    print(f"'LBW(Lora Block Weight)' is given, but the 'Inspire Pack' is not installed. The LBW= attribute is being ignored.")
                    model, clip = default_lora()
                else:
                    cls = nodes.NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                    model, clip, _ = cls().doit(model, clip, lora_name, model_weight, clip_weight, False, 0, lbw_a, lbw_b, "", lbw)
            else:
                model, clip = default_lora()
        else:
            print(f"LORA NOT FOUND: {orig_lora_name}")

    pass3 = [x.strip() for x in pass2.split("BREAK")]
    pass3 = [x for x in pass3 if x != '']

    if len(pass3) == 0:
        pass3 = ['']

    pass3_str = [f'[{x}]' for x in pass3]
    print(f"CLIP: {str.join(' + ', pass3_str)}")

    result = None

    for prompt in pass3:
        if clip_encoder is None:
            cur = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        else:
            cur = clip_encoder.encode(clip, prompt)[0]

        if result is not None:
            result = nodes.ConditioningConcat().concat(result, cur)[0]
        else:
            result = cur

    if processed is not None:
        processed.append(pass1)
        processed.append(pass2)
        processed.append(pass3)

    return model, clip, result


def starts_with_regex(pattern, text):
    regex = re.compile(pattern)
    return bool(regex.match(text))


def split_to_dict(text):
    pattern = r'\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)'
    matches = re.findall(pattern, text)

    result_dict = {key: value.strip() for key, value in matches}

    return result_dict


class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        if self.i >= len(self.items):
            self.i = 0
            if self.randomize_when_exhaust:
                random.shuffle(self.items)

        item = self.items[self.i]
        self.i += 1

        return item


class WildcardChooserDict:
    def __init__(self, items):
        self.items = items

    def get(self, seg):
        text = ""
        if 'ALL' in self.items:
            text = self.items['ALL']

        if seg.label in self.items:
            text += self.items[seg.label]

        return text


def split_string_with_sep(input_string):
    sep_pattern = r'\[SEP(?:\:\w+)?\]'

    substrings = re.split(sep_pattern, input_string)

    result_list = [None]
    matches = re.findall(sep_pattern, input_string)
    for i, substring in enumerate(substrings):
        result_list.append(substring)
        if i < len(matches):
            if matches[i] == '[SEP]':
                result_list.append(None)
            elif matches[i] == '[SEP:R]':
                result_list.append(random.randint(0, 1125899906842624))
            else:
                try:
                    seed = int(matches[i][5:-1])
                except:
                    seed = None
                result_list.append(seed)

    iterable = iter(result_list)
    return list(zip(iterable, iterable))


def process_wildcard_for_segs(wildcard):
    if wildcard.startswith('[LAB]'):
        raw_items = split_to_dict(wildcard)

        items = {}
        for k, v in raw_items.items():
            v = v.strip()
            if v != '':
                items[k] = v

        return 'LAB', WildcardChooserDict(items)

    elif starts_with_regex(r"\[(ASC|DSC|RND)\]", wildcard):
        mode = wildcard[1:4]
        items = split_string_with_sep(wildcard[5:])

        if mode == 'RND':
            random.shuffle(items)
            return mode, WildcardChooser(items, True)
        else:
            return mode, WildcardChooser(items, False)

    else:
        return None, WildcardChooser([(None, wildcard)], False)
