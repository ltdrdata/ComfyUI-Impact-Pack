import configparser
import os


version = "V3.20"

dependency_version = 9

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
config_path = os.path.join(my_path, "..", "..", "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")

MAX_RESOLUTION = 8192


def write_config():
    config = configparser.ConfigParser()
    config['default'] = {
        'dependency_version': dependency_version,
        'mmdet_skip': get_config()['mmdet_skip'],
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
                    'mmdet_skip': default_conf['mmdet_skip'].lower() == 'true'
               }

    except Exception:
        return {'dependency_version': 0, 'mmdet_skip': True}


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config
