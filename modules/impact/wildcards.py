import logging
import os
import random
import re
import threading

import folder_paths
import nodes
import numpy as np
import yaml
from impact import config, utils

wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wildcards"))

RE_WildCardQuantifier = re.compile(r"(?P<quantifier>\d+)#__(?P<keyword>[\w.\-+/*\\]+?)__", re.IGNORECASE)
wildcard_lock = threading.Lock()
wildcard_dict = {}

# Cache size limit in bytes (default: 50MB)
WILDCARD_CACHE_LIMIT = 50 * 1024 * 1024
# Flag to track if on-demand mode is active
_on_demand_mode = False

# Two-phase loading support
# available_wildcards: All discovered wildcard files (metadata only)
# loaded_wildcards: Actually loaded wildcard data
available_wildcards = {}  # key -> file_path mapping
loaded_wildcards = {}     # key -> loaded data


class LazyWildcardLoader:
    """
    Lazy loader for wildcard data to reduce memory usage.
    Acts as a list-like proxy that loads data on first access.
    """
    def __init__(self, file_path, file_type='txt'):
        self.file_path = file_path
        self.file_type = file_type
        self._data = None
        self._loaded = False

    def _load_txt(self):
        """Load .txt wildcard file"""
        try:
            with open(self.file_path, 'r', encoding="ISO-8859-1") as f:
                lines = f.read().splitlines()
                return [x for x in lines if x.strip() and not x.strip().startswith('#')]
        except (yaml.reader.ReaderError, UnicodeDecodeError):
            with open(self.file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                lines = f.read().splitlines()
                return [x for x in lines if x.strip() and not x.strip().startswith('#')]

    def _load_yaml(self):
        """Load .yaml/.yml wildcard file"""
        try:
            with open(self.file_path, 'r', encoding="ISO-8859-1") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except (yaml.reader.ReaderError, UnicodeDecodeError):
            with open(self.file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

    def get_data(self):
        """Get wildcard data, loading if necessary"""
        if not self._loaded:
            with wildcard_lock:
                if not self._loaded:  # Double-check locking
                    if self.file_type == 'txt':
                        self._data = self._load_txt()
                    elif self.file_type in ('yaml', 'yml'):
                        self._data = self._load_yaml()
                    self._loaded = True
        return self._data

    # List-like interface methods
    def __getitem__(self, index):
        """Support indexing like a list"""
        return self.get_data()[index]

    def __iter__(self):
        """Support iteration"""
        return iter(self.get_data())

    def __len__(self):
        """Support len() function"""
        return len(self.get_data())

    def __contains__(self, item):
        """Support 'in' operator"""
        return item in self.get_data()

    def __repr__(self):
        """String representation"""
        if self._loaded:
            return f"LazyWildcardLoader({self.file_path}, loaded={len(self._data)} items)"
        return f"LazyWildcardLoader({self.file_path}, not loaded)"

    def __bool__(self):
        """Support boolean evaluation"""
        return len(self.get_data()) > 0

    # Common list methods that may be used
    def count(self, value):
        """Count occurrences of value"""
        return self.get_data().count(value)

    def index(self, value, start=0, stop=None):
        """Find index of value"""
        if stop is None:
            return self.get_data().index(value, start)
        return self.get_data().index(value, start, stop)


def calculate_directory_size(directory_path, limit=None):
    """
    Calculate total size of all wildcard files in directory.

    Args:
        directory_path: Path to scan
        limit: Optional size limit in bytes. If provided, stops scanning immediately
               when total_size >= limit (for fast mode detection)

    Returns:
        Total size in bytes (or limit if exceeded)
    """
    total_size = 0
    try:
        for root, directories, files in os.walk(directory_path, followlinks=True):
            for file in files:
                if file.endswith(('.txt', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)

                        # Early termination: stop scanning when limit exceeded
                        if limit and total_size >= limit:
                            return total_size
                    except (OSError, FileNotFoundError):
                        pass
    except (OSError, FileNotFoundError):
        pass
    return total_size


def scan_wildcard_metadata(wildcard_path):
    """
    Scan directory for wildcard files and collect metadata only (no data loading).

    This is much faster than full loading for large wildcard collections.
    Only stores file paths in available_wildcards, actual data loaded on-demand.

    Args:
        wildcard_path: Directory to scan for wildcard files

    Returns:
        Number of wildcard files discovered
    """
    global available_wildcards

    discovered = 0
    try:
        for root, directories, files in os.walk(wildcard_path, followlinks=True):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, wildcard_path)
                    key = wildcard_normalize(os.path.splitext(rel_path)[0])
                    available_wildcards[key] = file_path
                    discovered += 1
                elif file.endswith('.yaml') or file.endswith('.yml'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, wildcard_path)
                    # YAML files are stored with their extension for proper loading
                    key_base = wildcard_normalize(os.path.splitext(rel_path)[0])
                    available_wildcards[key_base] = file_path
                    discovered += 1
    except (OSError, FileNotFoundError) as e:
        logging.warning(f"[Impact Pack] Error scanning wildcard directory {wildcard_path}: {e}")

    return discovered


def get_wildcard_list():
    """
    Get list of all available wildcards.

    Returns:
        - In full cache mode: all loaded wildcards
        - In on-demand mode: only loaded wildcards (same as get_loaded_wildcard_list)
    """
    with wildcard_lock:
        if _on_demand_mode:
            return [f"__{x}__" for x in loaded_wildcards.keys()]
        return [f"__{x}__" for x in wildcard_dict.keys()]


def get_loaded_wildcard_list():
    """
    Get list of actually loaded wildcards (on-demand mode only).

    Returns:
        List of wildcards that have been loaded into memory.
        In full cache mode, returns same as get_wildcard_list().
    """
    with wildcard_lock:
        if _on_demand_mode:
            return [f"__{x}__" for x in loaded_wildcards.keys()]
        return [f"__{x}__" for x in wildcard_dict.keys()]


def get_wildcard_dict():
    global wildcard_dict
    with wildcard_lock:
        return wildcard_dict


def find_wildcard_file(key):
    """
    Dynamically find a wildcard file by key (on-demand mode).

    For YAML files with nested structure (e.g., "colors/warm"):
    - Tries to find the parent YAML file (e.g., "colors.yaml")
    - Returns the YAML file path if found

    Searches in:
    1. Main wildcards directory
    2. Custom wildcards directory (if configured)

    Args:
        key: normalized wildcard key (e.g., "samples/flower", "colors/warm")

    Returns:
        Tuple of (file_path, is_yaml_nested) if found, (None, False) otherwise
    """
    # For YAML nested keys like "colors/warm", try parent file "colors.yaml"
    # Also try exact match for TXT files or top-level YAML keys

    # Case 1: Direct file match (TXT or top-level YAML)
    potential_paths = [
        f"{key}.txt",
        f"{key}.yaml",
        f"{key}.yml"
    ]

    for rel_path in potential_paths:
        file_path = os.path.join(wildcards_path, rel_path)
        if os.path.isfile(file_path):
            return (file_path, file_path.endswith(('.yaml', '.yml')))

    # Custom wildcards directory
    try:
        custom_path = config.get_config().get('custom_wildcards')
        if custom_path and os.path.exists(custom_path):
            for rel_path in potential_paths:
                file_path = os.path.join(custom_path, rel_path)
                if os.path.isfile(file_path):
                    return (file_path, file_path.endswith(('.yaml', '.yml')))
    except Exception:
        pass

    # Case 2: YAML nested key (e.g., "colors/warm" → "colors.yaml")
    if '/' in key:
        parent_key = key.split('/')[0]
        yaml_paths = [
            f"{parent_key}.yaml",
            f"{parent_key}.yml"
        ]

        for rel_path in yaml_paths:
            file_path = os.path.join(wildcards_path, rel_path)
            if os.path.isfile(file_path):
                return (file_path, True)

        # Custom wildcards directory
        try:
            custom_path = config.get_config().get('custom_wildcards')
            if custom_path and os.path.exists(custom_path):
                for rel_path in yaml_paths:
                    file_path = os.path.join(custom_path, rel_path)
                    if os.path.isfile(file_path):
                        return (file_path, True)
        except Exception:
            pass

    return (None, False)


def get_wildcard_value(key):
    """
    Get wildcard value from dictionary, automatically handling LazyWildcardLoader
    and on-demand loading.

    Args:
        key: wildcard key

    Returns:
        List of wildcard options (loaded if necessary), or None if not found
    """
    global loaded_wildcards

    # On-demand mode: dynamic file discovery and loading
    if _on_demand_mode:
        # Check if already loaded in cache (TXT on-demand or YAML pre-loaded)
        if key in loaded_wildcards:
            return loaded_wildcards[key]

        # Try to find and load TXT files dynamically
        # YAML files are already pre-loaded, so if not in cache, it doesn't exist
        file_path, is_yaml = find_wildcard_file(key)
        if file_path is None:
            # Fallback: Try pattern matching to find wildcards at any depth
            # Example: "dragon" matches "dragon.txt", "fantasy/dragon.txt", "dragon/fire.txt", etc.
            matched_keys = []
            for k in available_wildcards.keys():
                if (k == key or
                    k.endswith('/' + key) or
                    k.startswith(key + '/') or
                    ('/' + key + '/') in k):
                    matched_keys.append(k)

            if matched_keys:
                # Collect all options from matched keys
                all_options = []
                for matched_key in matched_keys:
                    # Load each matched wildcard
                    value = get_wildcard_value(matched_key)
                    if value:
                        all_options.extend(value)

                if all_options:
                    # Cache the combined result
                    loaded_wildcards[key] = all_options
                    logging.info(f"[Impact Pack] Wildcard '{key}' resolved via depth-agnostic pattern matching to {len(matched_keys)} keys: {matched_keys}")
                    return all_options

            return None

        # YAML files should already be loaded
        if is_yaml or file_path.endswith(('.yaml', '.yml')):
            # YAML was pre-loaded but key not found
            logging.warning(f"[Impact Pack] YAML wildcard '{key}' not found (pre-load issue)")
            return None

        # Load TXT file on-demand
        try:
            data = load_txt_wildcard(file_path)
            loaded_wildcards[key] = data
            logging.debug(f"[Impact Pack] Loaded TXT wildcard '{key}' on-demand from {file_path}")
            return data
        except Exception as e:
            logging.warning(f"[Impact Pack] Failed to load wildcard {key} from {file_path}: {e}")
            return None

    # Full cache mode or fallback: use wildcard_dict
    value = wildcard_dict.get(key)
    if isinstance(value, LazyWildcardLoader):
        return value.get_data()
    return value


def load_txt_wildcard(file_path):
    """Load a .txt wildcard file"""
    try:
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            lines = f.read().splitlines()
            return [x for x in lines if x.strip() and not x.strip().startswith('#')]
    except (yaml.reader.ReaderError, UnicodeDecodeError):
        with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
            lines = f.read().splitlines()
            return [x for x in lines if x.strip() and not x.strip().startswith('#')]


def load_yaml_wildcard(file_path, key_prefix=''):
    """Load a .yaml/.yml wildcard file and expand nested structures"""
    global loaded_wildcards

    try:
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    except (yaml.reader.ReaderError, UnicodeDecodeError):
        with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    if not yaml_data:
        return []

    # For nested YAML structures, expand into loaded_wildcards
    result = []
    for k, v in yaml_data.items():
        if isinstance(v, list):
            sub_key = wildcard_normalize(f"{key_prefix}/{k}") if key_prefix else wildcard_normalize(k)
            loaded_wildcards[sub_key] = v
            result.extend(v)
        elif isinstance(v, dict):
            # Recursive nested dict - register both parent and children keys
            # Collect all values from nested structure for parent key
            parent_key = wildcard_normalize(k)
            parent_values = []

            for k2, v2 in v.items():
                sub_key = wildcard_normalize(f"{k}/{k2}")
                if isinstance(v2, list):
                    loaded_wildcards[sub_key] = v2
                    parent_values.extend(v2)
                elif isinstance(v2, str):
                    loaded_wildcards[sub_key] = [v2]
                    parent_values.append(v2)
                elif isinstance(v2, (int, float)):
                    loaded_wildcards[sub_key] = [str(v2)]
                    parent_values.append(str(v2))

            # Register parent key with all child values
            if parent_values:
                loaded_wildcards[parent_key] = parent_values
                result.extend(parent_values)
        elif isinstance(v, str):
            sub_key = wildcard_normalize(f"{key_prefix}/{k}") if key_prefix else wildcard_normalize(k)
            loaded_wildcards[sub_key] = [v]
        elif isinstance(v, (int, float)):
            sub_key = wildcard_normalize(f"{key_prefix}/{k}") if key_prefix else wildcard_normalize(k)
            loaded_wildcards[sub_key] = [str(v)]

    return result if result else list(yaml_data.values())


def is_on_demand_mode():
    """Check if wildcards are running in on-demand mode"""
    return _on_demand_mode


def wildcard_normalize(x):
    return x.replace("\\", "/").replace(' ', '-').lower()


def read_wildcard(k, v, on_demand=False):
    """
    Read wildcard data with optional on-demand loading

    Args:
        k: wildcard key
        v: wildcard value (list, dict, str, or number)
        on_demand: if True, store LazyWildcardLoader instead of actual data
    """
    if isinstance(v, list):
        k = wildcard_normalize(k)
        wildcard_dict[k] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            new_key = f"{k}/{k2}"
            new_key = wildcard_normalize(new_key)
            read_wildcard(new_key, v2, on_demand)
    elif isinstance(v, str):
        k = wildcard_normalize(k)
        wildcard_dict[k] = [v]
    elif isinstance(v, (int, float)):
        k = wildcard_normalize(k)
        wildcard_dict[k] = [str(v)]

def read_wildcard_dict(wildcard_path, on_demand=False):
    """
    Read wildcard dictionary with optional on-demand loading

    Args:
        wildcard_path: path to wildcard directory
        on_demand: if True, use lazy loading to reduce memory usage

    Returns:
        wildcard_dict
    """
    global wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = wildcard_normalize(os.path.splitext(rel_path)[0])

                if on_demand:
                    # Store lazy loader instead of actual data
                    wildcard_dict[key] = LazyWildcardLoader(file_path, 'txt')
                else:
                    # Load data immediately (original behavior)
                    try:
                        with open(file_path, 'r', encoding="ISO-8859-1") as f:
                            lines = f.read().splitlines()
                            wildcard_dict[key] = [x for x in lines if x.strip() and not x.strip().startswith('#')]
                    except yaml.reader.ReaderError:
                        with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                            lines = f.read().splitlines()
                            wildcard_dict[key] = [x for x in lines if x.strip() and not x.strip().startswith('#')]
            elif file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)

                if on_demand:
                    # For YAML files in on-demand mode, we need to load and parse them
                    # since they may contain nested structures
                    loader = LazyWildcardLoader(file_path, 'yaml')
                    yaml_data = loader.get_data()
                    if yaml_data:
                        for k, v in yaml_data.items():
                            read_wildcard(k, v, on_demand)
                else:
                    # Load data immediately (original behavior)
                    try:
                        with open(file_path, 'r', encoding="ISO-8859-1") as f:
                            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                    except yaml.reader.ReaderError:
                        with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

                    for k, v in yaml_data.items():
                        read_wildcard(k, v, on_demand)

    return wildcard_dict


def process_comment_out(text):
    lines = text.split('\n')

    lines0 = []
    flag = False
    for line in lines:
        if line.lstrip().startswith('#'):
            flag = True
            continue

        if len(lines0) == 0:
            lines0.append(line)
        elif flag:
            lines0[-1] += ' ' + line
            flag = False
        else:
            lines0.append(line)

    return '\n'.join(lines0)


def process(text, seed=None):
    text = process_comment_out(text)

    if seed is not None:
        random.seed(seed)
    random_gen = np.random.default_rng(seed)

    local_wildcard_dict = get_wildcard_dict()

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
            wildcard_pattern = r"__([\w.\-+/*\\]+?)__"

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
                    else:
                        b = a

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    # Expand wildcard path or return the string after $$
                    def expand_wildcard_or_return_string(options, pattern, wildcard_pattern):
                        matches = re.findall(wildcard_pattern, pattern)
                        if len(options) == 1 and matches:
                            # $$<single wildcard>
                            return get_wildcard_options(pattern)
                        else:
                            # $$opt1|opt2|...
                            options[0] = pattern
                            return options

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options = expand_wildcard_or_return_string(options, multi_select_pattern[1], wildcard_pattern )
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options = expand_wildcard_or_return_string(options, multi_select_pattern[2], wildcard_pattern )

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1) if isinstance(option, str) else f"{option}".split('::', 1)

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
                def calculate_max(_options_length, _max_select_range):
                    return min(_max_select_range + 1, _options_length + 1) if _max_select_range > 0 else _options_length + 1

                def calculate_select_count(_max_value, _min_select_range, random_gen):
                    if max(_max_value, _min_select_range) <= 0:
                        return 0
                    # fix: low >= high
                    elif _max_value == _min_select_range:
                        return _max_value
                    else:
                        # fix: low >= high
                        _low_value = min(_min_select_range, _max_value)
                        _high_value = max(_min_select_range, _max_value)
                        return random_gen.integers(low=_low_value, high=_high_value, size=1)
                select_count = calculate_select_count(calculate_max(len(options), select_range[1]), select_range[0], random_gen)

            if select_count > len(options) or total_prob <= 1:
                random_gen.shuffle(options)
                selected_items = options
            else:
                selected_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)

            # x may be numpy.int32, convert to string
            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', str(x), count=1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'(?<!\\)\{((?:[^{}]|(?<=\\)[{}])*?)(?<!\\)\}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def get_wildcard_options(string):
        pattern = r"__([\w.\-+/*\\]+?)__"
        matches = re.findall(pattern, string)

        options = []

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)

            if '*' in keyword:
                logging.info(f"[Impact Pack] [get_wildcard_options] Processing wildcard pattern: keyword={keyword}")

            # Use get_wildcard_value for on-demand loading support
            wildcard_value = get_wildcard_value(keyword)

            if wildcard_value is not None:
                options.extend(wildcard_value)
            elif '*' in keyword:
                total_patterns = []
                found = False

                # For wildcard patterns, search through available wildcards
                search_dict = available_wildcards if _on_demand_mode else local_wildcard_dict

                # Special case: __*/name__ should match both 'name' and 'name/*' at any depth
                if keyword.startswith('*/') and len(keyword) > 2:
                    base_name = keyword[2:]  # Remove '*/' prefix

                    logging.info(f"[Impact Pack] [get_wildcard_options] Pattern: keyword={keyword}, base={base_name}, on_demand={_on_demand_mode}, search_dict_size={len(search_dict)}")

                    matched_count = 0
                    for k in search_dict.keys():
                        # Match if key ends with base_name or contains base_name/subdirs
                        # Pattern matching examples for base_name="dragon":
                        #   "dragon" -> match (exact)
                        #   "fantasy/dragon" -> match (nested file)
                        #   "dragon/fire" -> match (subfolder)
                        #   "fantasy/dragon/fire" -> match (deeply nested)
                        if (k == base_name or
                            k.endswith('/' + base_name) or
                            k.startswith(base_name + '/') or
                            ('/' + base_name + '/') in k):
                            logging.info(f"[Impact Pack] [get_wildcard_options] Matched: {k}")
                            v = get_wildcard_value(k)
                            if v:
                                total_patterns += v
                                found = True
                                matched_count += 1

                    logging.info(f"[Impact Pack] [get_wildcard_options] Result: matched={matched_count}, patterns={len(total_patterns)}")
                else:
                    # General wildcard pattern matching
                    subpattern = keyword.replace('*', '.*').replace('+', '\\+')
                    for k in search_dict.keys():
                        if re.match(subpattern, k) is not None or re.match(subpattern, k+'/') is not None:
                            # Load on-demand if needed
                            v = get_wildcard_value(k)
                            if v:
                                total_patterns += v
                                found = True

                if found:
                    options.extend(total_patterns)
            # Note: Fallback to __*/name__ is handled in replace_wildcard, not here

        return options

    def replace_wildcard(string):
        pattern = r"__([\w.\-+/*\\]+?)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)

            # Use get_wildcard_value for on-demand loading support
            options = get_wildcard_value(keyword)

            if options is not None:
                # look for adjusted probability
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
                selected_item = random_gen.choice(options, p=normalized_probabilities, replace=False)
                replacement = re.sub(r'^\s*[0-9.]+::', '', selected_item, count=1)
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                total_patterns = []
                found = False

                # For wildcard patterns, search through available wildcards
                search_dict = available_wildcards if _on_demand_mode else local_wildcard_dict

                # Special case: __*/name__ should match both 'name' and 'name/*' at any depth
                if keyword.startswith('*/') and len(keyword) > 2:
                    base_name = keyword[2:]  # Remove '*/' prefix

                    for k in search_dict.keys():
                        # Match if key ends with base_name or contains base_name/subdirs
                        # Pattern matching examples for base_name="dragon":
                        #   "dragon" -> match (exact)
                        #   "fantasy/dragon" -> match (nested file)
                        #   "dragon/fire" -> match (subfolder)
                        #   "fantasy/dragon/fire" -> match (deeply nested)
                        if (k == base_name or
                            k.endswith('/' + base_name) or
                            k.startswith(base_name + '/') or
                            ('/' + base_name + '/') in k):
                            v = get_wildcard_value(k)
                            if v:
                                total_patterns += v
                                found = True
                else:
                    # General wildcard pattern matching
                    subpattern = keyword.replace('*', '.*').replace('+', '\\+')
                    for k in search_dict.keys():
                        if re.match(subpattern, k) is not None or re.match(subpattern, k+'/') is not None:
                            # Load on-demand if needed
                            v = get_wildcard_value(k)
                            if v:
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
    return re.match(r'^-?(\d*\.?\d+|\d+\.?\d*)$', input_str) is not None


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
        loader = None

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
                elif sub_item.startswith("LOADER="):
                    loader = sub_item[7:]

        if a is None:
            a = 1.0
        if b is None:
            b = a

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b, loader))
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

    return None


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

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b, loader in loras:
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
            logging.info(f"LOAD LORA: {lora_name}: {model_weight}, {clip_weight}, LBW={lbw}, A={lbw_a}, B={lbw_b}, LOADER={loader}")

            if loader is not None:
                if loader == 'nunchaku':
                    if 'NunchakuFluxLoraLoader' not in nodes.NODE_CLASS_MAPPINGS:
                        logging.warning("To use `LOADER=nunchaku`, 'ComfyUI-nunchaku' is required. The LOADER= attribute is being ignored.")
                    cls = nodes.NODE_CLASS_MAPPINGS['NunchakuFluxLoraLoader']
                    model = cls().load_lora(model, lora_name, model_weight)[0]
                else:
                    logging.warning(f"LORA LOADER NOT FOUND: '{loader}'")
            else:
                def default_lora():
                    return nodes.LoraLoader().load_lora(model, clip, lora_name, model_weight, clip_weight)

                if lbw is not None:
                    if 'LoraLoaderBlockWeight //Inspire' not in nodes.NODE_CLASS_MAPPINGS:
                        utils.try_install_custom_node(
                            'https://github.com/ltdrdata/ComfyUI-Inspire-Pack',
                            "To use 'LBW=' syntax in wildcards, 'Inspire Pack' extension is required.")

                        logging.warning("'LBW(Lora Block Weight)' is given, but the 'Inspire Pack' is not installed. The LBW= attribute is being ignored.")
                        model, clip = default_lora()
                    else:
                        cls = nodes.NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                        model, clip, _ = cls().doit(model, clip, lora_name, model_weight, clip_weight, False, 0, lbw_a, lbw_b, "", lbw)

                else:
                    model, clip = default_lora()
        else:
            logging.warning(f"LORA NOT FOUND: {orig_lora_name}")

    pass3 = [x.strip() for x in pass2.split("BREAK")]
    pass3 = [x for x in pass3 if x != '']

    if len(pass3) == 0:
        pass3 = ['']

    pass3_str = [f'[{x}]' for x in pass3]
    logging.info(f"CLIP: {str.join(' + ', pass3_str)}")

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
    return regex.match(text)


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
                except Exception:
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

    else:
        match = starts_with_regex(r"\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND)\]", wildcard)

        if match:
            mode = match[1]
            items = split_string_with_sep(wildcard[len(match[0]):])

            if mode == 'RND':
                random.shuffle(items)
                return mode, WildcardChooser(items, True)
            else:
                return mode, WildcardChooser(items, False)

        else:
            return None, WildcardChooser([(None, wildcard)], False)


def load_yaml_files_only(wildcard_path):
    """
    Load only YAML wildcard files from a directory (for on-demand mode).

    YAML files must be pre-loaded because wildcard keys are inside the file contents.
    Unlike TXT files where "samples/flower.txt" → "__samples/flower__" (file path = key),
    YAML files like "colors.yaml" can contain multiple keys (colors/warm, colors/cold, etc.)
    that are only discoverable by parsing the entire file content.

    Example:
        colors.yaml:
            warm: [red, orange, yellow]   → __colors/warm__
            cold: [blue, green, purple]   → __colors/cold__

        To know that "colors/warm" exists, we must parse colors.yaml completely.
        Therefore, YAML files cannot be truly on-demand loaded.

    Args:
        wildcard_path: Directory to scan for YAML files

    Returns:
        Number of YAML wildcard files loaded (not keys)
    """
    global loaded_wildcards

    yaml_count = 0
    try:
        for root, directories, files in os.walk(wildcard_path, followlinks=True):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    file_path = os.path.join(root, file)
                    try:
                        # Load YAML file and register all sub-keys
                        load_yaml_wildcard(file_path, key_prefix='')
                        yaml_count += 1
                        logging.debug(f"[Impact Pack] Pre-loaded YAML file: {file_path}")
                    except Exception as e:
                        logging.warning(f"[Impact Pack] Failed to load YAML file {file_path}: {e}")
    except (OSError, FileNotFoundError) as e:
        logging.warning(f"[Impact Pack] Error scanning YAML files in {wildcard_path}: {e}")

    return yaml_count


def get_cache_limit():
    """Get cache limit from config or use default"""
    try:
        cfg = config.get_config()
        if 'wildcard_cache_limit_mb' in cfg:
            return cfg['wildcard_cache_limit_mb'] * 1024 * 1024  # Convert MB to bytes
    except Exception:
        pass
    return WILDCARD_CACHE_LIMIT


def wildcard_load():
    """
    Load wildcards with automatic on-demand mode when total size exceeds limit.

    If total wildcard file size < cache_limit (default 50MB):
        - Full cache mode: all data loaded into memory (original behavior)
    If total wildcard file size >= cache_limit:
        - On-demand mode: TXT files loaded dynamically when accessed
        - YAML files always pre-loaded immediately (limitation)

    YAML Limitation:
        YAML wildcards must be pre-loaded because wildcard keys are embedded
        inside the file contents, not in the file path.

        TXT files:  "samples/flower.txt" → key is "__samples/flower__" (file path = key)
        YAML files: "colors.yaml" contains:
                        warm: [red, orange]     → key is "__colors/warm__"
                        cold: [blue, green]     → key is "__colors/cold__"

        To discover that "colors/warm" exists, we must parse colors.yaml completely.
        Therefore, YAML files cannot be truly on-demand loaded and are pre-loaded at startup.
    """
    global wildcard_dict, available_wildcards, loaded_wildcards, _on_demand_mode
    wildcard_dict = {}
    available_wildcards = {}
    loaded_wildcards = {}
    _on_demand_mode = False

    with wildcard_lock:
        # Calculate total size of wildcard files (with early termination)
        cache_limit = get_cache_limit()
        total_size = calculate_directory_size(wildcards_path, limit=cache_limit)

        # Add custom wildcards directory size if it exists
        custom_wildcards_path = None
        try:
            custom_wildcards_path = config.get_config().get('custom_wildcards')
            if custom_wildcards_path and os.path.exists(custom_wildcards_path):
                # Early termination: if already exceeded, don't scan custom dir
                if total_size < cache_limit:
                    custom_size = calculate_directory_size(custom_wildcards_path,
                                                          limit=cache_limit - total_size)
                    total_size += custom_size
        except Exception:
            pass

        # Determine loading mode based on total size
        if total_size >= cache_limit:
            _on_demand_mode = True
            logging.info(f"[Impact Pack] Wildcard total size ({total_size / (1024*1024):.2f} MB) "
                        f"exceeds cache limit ({cache_limit / (1024*1024):.2f} MB). "
                        f"Using on-demand loading mode (TXT files loaded dynamically).")

            # On-demand mode: Scan for TXT file metadata and load YAML files immediately
            # Metadata scan discovers TXT files without loading their content
            txt_count = scan_wildcard_metadata(wildcards_path)
            if custom_wildcards_path and os.path.exists(custom_wildcards_path):
                txt_count += scan_wildcard_metadata(custom_wildcards_path)

            # Load YAML files immediately (limitation: YAML keys are inside file content)
            yaml_count = load_yaml_files_only(wildcards_path)
            if custom_wildcards_path and os.path.exists(custom_wildcards_path):
                yaml_count += load_yaml_files_only(custom_wildcards_path)

            logging.info(f"[Impact Pack] On-demand mode active. "
                        f"Discovered {txt_count} TXT wildcards (metadata only). "
                        f"Pre-loaded {yaml_count} YAML wildcards. "
                        f"TXT wildcard content will be loaded only when accessed.")
        else:
            logging.info(f"[Impact Pack] Wildcard total size ({total_size / (1024*1024):.2f} MB) "
                        f"is within cache limit ({cache_limit / (1024*1024):.2f} MB). "
                        f"Using full cache mode.")

            # Full cache mode: load all data immediately (original behavior)
            read_wildcard_dict(wildcards_path, on_demand=False)

            try:
                if custom_wildcards_path:
                    read_wildcard_dict(custom_wildcards_path, on_demand=False)
            except Exception:
                logging.info("[Impact Pack] Failed to load custom wildcards directory.")

        logging.info("[Impact Pack] Wildcards loading done.")
