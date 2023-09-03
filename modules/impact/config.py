import configparser
import os


version = "V4.0"

dependency_version = 11

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
                            'sam_editor_model': get_config()['sam_editor_model']
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
                    'sam_editor_model': 'sam_vit_b_01ec64.pth'
               }

    except Exception:
        return {'dependency_version': 0, 'mmdet_skip': True, 'sam_editor_cpu': False, 'sam_editor_model': 'sam_vit_b_01ec64.pth'}


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config
