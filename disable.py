import os
import sys
import time
import platform
import shutil
import subprocess

comfy_path = '../..'

def rmtree(path):
    retry_count = 3

    while True:
        try:
            retry_count -= 1

            if platform.system() == "Windows":
                subprocess.check_call(['attrib', '-R', path + '\\*', '/S'])

            shutil.rmtree(path)

            return True

        except Exception as ex:
            print(f"ex: {ex}")
            time.sleep(3)

            if retry_count < 0:
                raise ex

            print(f"Uninstall retry({retry_count})")

js_dest_path = os.path.join(comfy_path, "web", "extensions", "impact-pack")

if os.path.exists(js_dest_path):
    rmtree(js_dest_path)
        

