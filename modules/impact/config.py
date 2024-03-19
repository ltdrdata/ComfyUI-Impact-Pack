import configparser
import os


version_code = [4, 84]
version = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')

dependency_version = 20

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
config_path = os.path.join(my_path, "..", "..", "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")

MAX_RESOLUTION = 8192


def write_config():
    config = configparser.ConfigParser()
    config['default'] = {
                            'dependency_version': str(dependency_version),
                            'mmdet_skip': str(get_config()['mmdet_skip']),
                            'sam_editor_cpu': str(get_config()['sam_editor_cpu']),
                            'sam_editor_model': get_config()['sam_editor_model'],
                            'custom_wildcards': get_config()['custom_wildcards'],
                            'disable_gpu_opencv': get_config()['disable_gpu_opencv'],
                        }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        return {
                    'dependency_version': int(default_conf['dependency_version']),
                    'mmdet_skip': default_conf['mmdet_skip'].lower() == 'true' if 'mmdet_skip' in default_conf else True,
                    'sam_editor_cpu': default_conf['sam_editor_cpu'].lower() == 'true' if 'sam_editor_cpu' in default_conf else False,
                    'sam_editor_model': 'sam_vit_b_01ec64.pth',
                    'custom_wildcards': default_conf['custom_wildcards'] if 'custom_wildcards' in default_conf else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
                    'disable_gpu_opencv': default_conf['disable_gpu_opencv'].lower() == 'true' if 'disable_gpu_opencv' in default_conf else True
               }

    except Exception:
        return {
            'dependency_version': 0,
            'mmdet_skip': True,
            'sam_editor_cpu': False,
            'sam_editor_model': 'sam_vit_b_01ec64.pth',
            'custom_wildcards': os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
            'disable_gpu_opencv': True
        }


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config
