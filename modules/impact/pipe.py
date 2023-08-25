import folder_paths

class ToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "bbox_detector": ("BBOX_DETECTOR", ),
                     "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                     "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL",),
                    "segm_detector_opt": ("SEGM_DETECTOR",),
                    "detailer_hook": ("DETAILER_HOOK",),
                }}

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        pipe = (kwargs['model'], kwargs['clip'], kwargs['vae'], kwargs['positive'], kwargs['negative'], kwargs['wildcard'], kwargs['bbox_detector'],
                kwargs.get('segm_detector_opt', None), kwargs.get('sam_model_opt', None), kwargs.get('detailer_hook', None))
        return (pipe, )


class FromDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL", "SEGM_DETECTOR", "DETAILER_HOOK")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative", "bbox_detector", "sam_model_opt", "segm_detector_opt", "detailer_hook")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook = detailer_pipe
        return model, clip, vae, positive, negative, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook


class FromDetailerPipe_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }, }

    RETURN_TYPES = ("DETAILER_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL", "SEGM_DETECTOR", "DETAILER_HOOK")
    RETURN_NAMES = ("detailer_pipe", "model", "clip", "vae", "positive", "negative", "bbox_detector", "sam_model_opt", "segm_detector_opt", "detailer_hook")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook = detailer_pipe
        return detailer_pipe, model, clip, vae, positive, negative, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook


class ToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     },
                }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, model, clip, vae, positive, negative):
        pipe = (model, clip, vae, positive, negative)
        return (pipe, )


class FromBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe):
        model, clip, vae, positive, negative = basic_pipe
        return model, clip, vae, positive, negative


class FromBasicPipe_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE",), }, }

    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("basic_pipe", "model", "clip", "vae", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe):
        model, clip, vae, positive, negative = basic_pipe
        return basic_pipe, model, clip, vae, positive, negative


class BasicPipeToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE",),
                             "bbox_detector": ("BBOX_DETECTOR", ),
                             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                             "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                             },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                    "segm_detector_opt": ("SEGM_DETECTOR",),
                    "detailer_hook": ("DETAILER_HOOK",),
                    },
                }

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        basic_pipe = kwargs['basic_pipe']
        bbox_detector = kwargs['bbox_detector']
        wildcard = kwargs['wildcard']
        sam_model_opt = kwargs.get('sam_model_opt', None)
        segm_detector_opt = kwargs.get('segm_detector_opt', None)
        detailer_hook = kwargs.get('detailer_hook', None)

        model, clip, vae, positive, negative = basic_pipe
        pipe = model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook
        return (pipe, )


class DetailerPipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }}

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, clip, vae, positive, negative, _, _, _, _, _ = detailer_pipe
        pipe = model, clip, vae, positive, negative
        return (pipe, )


class EditBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"basic_pipe": ("BASIC_PIPE",), },
                "optional": {
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     },
                }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe, model=None, clip=None, vae=None, positive=None, negative=None):
        res_model, res_clip, res_vae, res_positive, res_negative = basic_pipe

        if model is not None:
            res_model = model

        if clip is not None:
            res_clip = clip

        if vae is not None:
            res_vae = vae

        if positive is not None:
            res_positive = positive

        if negative is not None:
            res_negative = negative

        pipe = res_model, res_clip, res_vae, res_positive, res_negative

        return (pipe, )


class EditDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detailer_pipe": ("DETAILER_PIPE",),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "sam_model": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "detailer_hook": ("DETAILER_HOOK",),
            },
        }

    RETURN_TYPES = ("DETAILER_PIPE",)
    RETURN_NAMES = ("detailer_pipe",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, *args, **kwargs):
        detailer_pipe = kwargs['detailer_pipe']
        wildcard = kwargs['wildcard']
        model = kwargs.get('model', None)
        clip = kwargs.get('clip', None)
        vae = kwargs.get('vae', None)
        positive = kwargs.get('positive', None)
        negative = kwargs.get('negative', None)
        bbox_detector = kwargs.get('bbox_detector', None)
        sam_model = kwargs.get('sam_model', None)
        segm_detector = kwargs.get('segm_detector', None)
        detailer_hook = kwargs.get('detailer_hook', None)

        res_model, res_clip, res_vae, res_positive, res_negative, res_wildcard, res_bbox_detector, res_segm_detector, res_sam_model, res_detailer_hook = detailer_pipe

        if model is not None:
            res_model = model

        if clip is not None:
            res_clip = clip

        if vae is not None:
            res_vae = vae

        if positive is not None:
            res_positive = positive

        if negative is not None:
            res_negative = negative

        if bbox_detector is not None:
            res_bbox_detector = bbox_detector

        if segm_detector is not None:
            res_segm_detector = segm_detector

        if wildcard != "":
            res_wildcard = wildcard

        if sam_model is not None:
            res_sam_model = sam_model

        if detailer_hook is not None:
            res_detailer_hook = detailer_hook

        pipe = res_model, res_clip, res_vae, res_positive, res_negative, res_wildcard, res_bbox_detector, res_segm_detector, res_sam_model, res_detailer_hook

        return (pipe, )
