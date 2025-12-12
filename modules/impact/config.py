import configparser
import logging
import os

version_code = [8, 28]
version = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
config_path = os.path.join(my_path, "..", "..", "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")


def write_config():
    config = configparser.ConfigParser()
    config['default'] = {
                            'sam_editor_cpu': str(get_config()['sam_editor_cpu']),
                            'sam_editor_model': get_config()['sam_editor_model'],
                            'custom_wildcards': get_config()['custom_wildcards'],
                            'disable_gpu_opencv': get_config()['disable_gpu_opencv'],
                            'wildcard_cache_limit_mb': str(get_config()['wildcard_cache_limit_mb']),
                        }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        # Strip quotes from custom_wildcards path if present
        custom_wildcards_path = default_conf.get('custom_wildcards', '').strip('\'"')

        if not os.path.exists(custom_wildcards_path):
            logging.warning(f"[Impact Pack] custom_wildcards path not found: {custom_wildcards_path}. Using default path.")
            custom_wildcards_path = os.path.join(my_path, "..", "..", "custom_wildcards")

        default_conf['custom_wildcards'] = custom_wildcards_path

        # Parse wildcard_cache_limit_mb with default value of 50MB
        cache_limit_mb = 50
        if 'wildcard_cache_limit_mb' in default_conf:
            try:
                cache_limit_mb = float(default_conf['wildcard_cache_limit_mb'])
            except ValueError:
                logging.warning(f"[Impact Pack] Invalid wildcard_cache_limit_mb value: {default_conf['wildcard_cache_limit_mb']}. Using default: 50")
                cache_limit_mb = 50

        return {
                    'sam_editor_cpu': default_conf['sam_editor_cpu'].lower() == 'true' if 'sam_editor_cpu' in default_conf else False,
                    'sam_editor_model': default_conf['sam_editor_model'].lower() if 'sam_editor_model' else 'sam_vit_b_01ec64.pth',
                    'custom_wildcards': default_conf['custom_wildcards'] if 'custom_wildcards' in default_conf else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
                    'disable_gpu_opencv': default_conf['disable_gpu_opencv'].lower() == 'true' if 'disable_gpu_opencv' in default_conf else True,
                    'wildcard_cache_limit_mb': cache_limit_mb
               }

    except Exception:
        return {
            'sam_editor_cpu': False,
            'sam_editor_model': 'sam_vit_b_01ec64.pth',
            'custom_wildcards': os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
            'disable_gpu_opencv': True,
            'wildcard_cache_limit_mb': 50
        }


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config
