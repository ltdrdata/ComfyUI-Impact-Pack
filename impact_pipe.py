class ToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL",),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "bbox_detector": ("BBOX_DETECTOR", ),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                }}

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, model, vae, positive, negative, bbox_detector, sam_model_opt=None):
        pipe = (model, vae, positive, negative, bbox_detector, sam_model_opt)
        return (pipe, )


class FromDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",), }, }

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "BBOX_DETECTOR", "SAM_MODEL")
    RETURN_NAMES = ("model", "vae", "positive", "negative", "bbox_detector", "sam_model_opt")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe):
        model, vae, positive, negative, bbox_detector, sam_model_opt = detailer_pipe
        return model, vae, positive, negative, bbox_detector, sam_model_opt


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


class BasicPipeToDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"basic_pipe": ("BASIC_PIPE",),
                             "bbox_detector": ("BBOX_DETECTOR", ), },
                "optional": {"sam_model_opt": ("SAM_MODEL", ), },
                }

    RETURN_TYPES = ("DETAILER_PIPE", )
    RETURN_NAMES = ("detailer_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, basic_pipe, bbox_detector, sam_model_opt=None):
        model, _, vae, positive, negative = basic_pipe
        pipe = model, vae, positive, negative, bbox_detector, sam_model_opt
        return (pipe, )


class DetailerPipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"detailer_pipe": ("DETAILER_PIPE",),
                             "clip": ("CLIP",), }, }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe, clip):
        model, vae, positive, negative, _, _ = detailer_pipe
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
            "required": {"detailer_pipe": ("DETAILER_PIPE",), },
            "optional": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "sam_model": ("SAM_MODEL",), },
        }

    RETURN_TYPES = ("DETAILER_PIPE",)
    RETURN_NAMES = ("detailer_pipe",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, detailer_pipe, model=None, vae=None, positive=None, negative=None, bbox_detector=None, sam_model=None):
        res_model, res_vae, res_positive, res_negative, res_bbox_detector, res_sam_model = detailer_pipe

        if model is not None:
            res_model = model

        if vae is not None:
            res_vae = vae

        if positive is not None:
            res_positive = positive

        if negative is not None:
            res_negative = negative

        if bbox_detector is not None:
            res_bbox_detector = bbox_detector

        if sam_model is not None:
            res_sam_model = sam_model

        pipe = res_model, res_vae, res_positive, res_negative, res_bbox_detector, res_sam_model

        return (pipe, )
