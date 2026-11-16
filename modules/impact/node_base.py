from comfy_api.latest import io


class ImpactNode(io.ComfyNode):
    """A V3 node that can still be called the V1 way, `Node().doit(...)`, as other node packs do."""

    @classmethod
    def doit(cls, *args, **kwargs):
        return cls.execute(*args, **kwargs).result
