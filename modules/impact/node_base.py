import logging

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from server import PromptServer


class ImpactNode(io.ComfyNode):
    """A V3 node that can still be called the V1 way, `Node().doit(...)`, as other node packs do."""

    @classmethod
    def doit(cls, *args, **kwargs):
        return cls.execute(*args, **kwargs).result


def warn_on_node(name, text):
    """Says in the log and on the node that is running, in the page, that the node did something else than what its inputs ask for."""
    logging.warning(f"[Impact Pack] {name}: {text}")
    context = get_executing_context()
    if context is not None:
        PromptServer.instance.send_sync("impact-node-warning", {"node_id": context.node_id, "text": text}, PromptServer.instance.client_id)
