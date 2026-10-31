import os
import re
import random
import nodes
from aiohttp import web
from server import PromptServer
from impact import config, wildcards


_LORA_RE = re.compile(r"<lora:([^>]+)>")


def achar_arquivo_wildcard(nome_arquivo):
    if "/" in nome_arquivo or "\\" in nome_arquivo:
        return None
    root = os.path.realpath(config.get_config()["custom_wildcards"])
    path = os.path.realpath(os.path.join(root, nome_arquivo))
    if os.path.commonpath((root, path)) == root and os.path.isfile(path):
        return path
    return None


def parse_linhas_wildcard(caminho):
    if not caminho:
        return []
    entries = []
    counts = {}
    for line in wildcards.load_txt_wildcard(caminho):
        loras = wildcards.extract_lora_values(line)
        if not loras or not loras[0][0]:
            continue
        name = loras[0][0].strip()
        count = counts.get(name, 0)
        label = f"{name} ({count})" if count else name
        counts[name] = count + 1
        entries.append((label, line.strip(), name))
    return entries


def listar_categorias_wildcard():
    root = config.get_config()["custom_wildcards"]
    categories = sorted(os.path.splitext(name)[0] for name in os.listdir(root)
                        if name.endswith(".txt") and achar_arquivo_wildcard(name))
    return categories or ["styles", "poses"]


def entry_with_strength(entry, strength):
    def replace(match):
        parts = match.group(1).split(":")
        name = parts.pop(0).strip()
        for _ in range(2):
            if parts and wildcards.is_numeric_string(parts[0]):
                parts.pop(0)
        return "<lora:" + ":".join([name, str(strength), str(strength), *parts]) + ">"
    return _LORA_RE.sub(replace, entry, count=1)


def nome_arquivo_limpo(nome, fallback="no_lora"):
    name = nome.replace("\\", "/").split("/")[-1]
    name = re.sub(r'[<>:"/\\|?*\[\],]', "", name)
    return re.sub(r"\s+", "_", name).strip("_") or fallback


def clean_prompt(text):
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s+", " ", text).strip().strip(",").strip()


class _PromptEncoder:
    def encode(self, clip, text):
        return nodes.CLIPTextEncode().encode(clip, clean_prompt(text))


@PromptServer.instance.routes.get("/impact/get_items")
async def get_items_handler(request):
    category = request.query.get("category", request.query.get("categoria", ""))
    entries = parse_linhas_wildcard(achar_arquivo_wildcard(f"{category}.txt"))
    return web.json_response([entry[0] for entry in entries])


class LoraWildcardSource:
    @classmethod
    def INPUT_TYPES(cls):
        # Scan wildcard text files from custom_wildcards directory dynamically
        categorias = listar_categorias_wildcard()
        primeira_categoria = categorias[0] if categorias else "styles"

        # Default fallback to first available category file list
        caminho = achar_arquivo_wildcard(primeira_categoria + ".txt")
        entradas = parse_linhas_wildcard(caminho)
        labels_padrao = [e[0] for e in entradas] or [f"({primeira_categoria}.txt not found)"]

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "category": (categorias,),
                "lora_wildcard": (labels_padrao,),
                "mode": (["fixed", "random", "sequential", "batch"],),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, very aesthetic, absurdres, 1girl",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "conditioning", "text", "lora_name", "name")
    OUTPUT_IS_LIST = (True, True, True, True, True, True)
    FUNCTION = "build"
    CATEGORY = "Impact Pack/Wildcard"
    DESCRIPTION = ("Generic LoRA wildcard source: fixed / random / sequential / batch. "
                   "Select category (wildcard text file) and the corresponding LoRA wildcard. "
                   "The node automatically applies the LoRAs and encodes the prompt.")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def build(self, model, clip, category, lora_wildcard, mode, strength, base_prompt, seed):
        entries = parse_linhas_wildcard(achar_arquivo_wildcard(f"{category}.txt"))
        if not entries:
            selected = [(f"no_{category}", "", "no_lora")]
        elif mode == "batch":
            selected = entries
        elif mode == "random":
            selected = [random.Random(seed).choice(entries)]
        elif mode == "sequential":
            selected = [entries[seed % len(entries)]]
        else:
            entry = next((entry for entry in entries if entry[0] == lora_wildcard), None)
            if entry is None:
                raise ValueError(f"LoRA wildcard '{lora_wildcard}' not found in category '{category}'")
            selected = [entry]

        outputs = tuple([] for _ in self.RETURN_TYPES)
        for label, entry, name in selected:
            prompt = ", ".join(part.strip().strip(",") for part in
                               (base_prompt, entry_with_strength(entry, strength)) if part.strip())
            processed = []
            result_model, result_clip, conditioning = wildcards.process_with_loras(
                prompt, model, clip, clip_encoder=_PromptEncoder(), seed=seed, processed=processed)
            values = (result_model, result_clip, conditioning, clean_prompt(processed[1]), nome_arquivo_limpo(name), label)
            for output, value in zip(outputs, values):
                output.append(value)
        return outputs
