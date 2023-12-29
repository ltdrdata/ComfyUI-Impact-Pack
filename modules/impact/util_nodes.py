from impact.utils import any_typ, ByPassTypeTuple
import comfy_extras.nodes_mask
from nodes import MAX_RESOLUTION

class GeneralSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
                    "sel_mode": ("BOOLEAN", {"default": True, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False}),
                    },
                "optional": {
                    "input1": (any_typ,),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = (any_typ, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        selected_label = input_name
        node_id = kwargs['unique_id']
        nodelist = kwargs['extra_pnginfo']['workflow']['nodes']
        for node in nodelist:
            if str(node['id']) == node_id:
                inputs = node['inputs']

                for slot in inputs:
                    if slot['name'] == input_name and 'label' in slot:
                        selected_label = slot['label']

                break

        if input_name in kwargs:
            return (kwargs[input_name], selected_label, selected_index)
        else:
            print(f"ImpactSwitch: invalid select index (ignored)")
            return (None, "", selected_index)


class GeneralInversedSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
                    "input": (any_typ,),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ByPassTypeTuple((any_typ, ))
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, input, unique_id):
        res = []

        for i in range(0, select):
            if select == i+1:
                res.append(input)
            else:
                res.append(None)

        return res


class ImageMaskSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            "images1": ("IMAGE",),
        },

            "optional": {
                "mask1_opt": ("MASK",),
                "images2_opt": ("IMAGE",),
                "mask2_opt": ("MASK",),
                "images3_opt": ("IMAGE",),
                "mask3_opt": ("MASK",),
                "images4_opt": ("IMAGE",),
                "mask4_opt": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, images1, mask1_opt=None, images2_opt=None, mask2_opt=None, images3_opt=None, mask3_opt=None,
             images4_opt=None, mask4_opt=None):
        if select == 1:
            return images1, mask1_opt,
        elif select == 2:
            return images2_opt, mask2_opt,
        elif select == 3:
            return images3_opt, mask3_opt,
        else:
            return images4_opt, mask4_opt,


class RemoveNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, samples):
        res = {key: value for key, value in samples.items() if key != 'noise_mask'}
        return (res, )


class ImagePasteMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, destination, source, x, y, resize_source, mask = None):
        destination = destination.clone().movedim(-1, 1)
        output = comfy_extras.nodes_mask.composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return (output,)


from impact.utils import any_typ

class ImpactLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "data": (any_typ, ""),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    CATEGORY = "ImpactPack/Debug"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "doit"

    def doit(self, data, prompt, extra_pnginfo):
        shape = ""
        if hasattr(data, "shape"):
            shape = f"{data.shape} / "

        print(f"[IMPACT LOGGER]: {shape}{data}")

        print(f"         PROMPT: {prompt}")

        # for x in prompt:
        #     if 'inputs' in x and 'populated_text' in x['inputs']:
        #         print(f"PROMP: {x['10']['inputs']['populated_text']}")
        #
        # for x in extra_pnginfo['workflow']['nodes']:
        #     if x['type'] == 'ImpactWildcardProcessor':
        #         print(f" WV : {x['widgets_values'][1]}\n")

        return {}


class ImpactDummyInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    CATEGORY = "ImpactPack/Debug"

    RETURN_TYPES = (any_typ,)
    FUNCTION = "doit"

    def doit(self):
        return ("DUMMY",)
