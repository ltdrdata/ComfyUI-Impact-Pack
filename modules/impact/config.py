import configparser
import os

version = "V2.21.2"

dependency_version = 1

my_path = os.path.dirname(__file__)
config_path = os.path.join(my_path, "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")

MAX_RESOLUTION = 8192

def write_config(comfy_path):
    config = configparser.ConfigParser()
    config['default'] = {
        'dependency_version': dependency_version,
        'comfy_path': comfy_path
    }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        return default_conf['comfy_path'], int(default_conf['dependency_version'])
    except Exception:
        return "", 0
