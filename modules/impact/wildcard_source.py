import os
import re
import random
import logging
import folder_paths
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
        loras, _ = extrair_e_limpar_loras(line)
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
    try:
        categories = sorted(os.path.splitext(name)[0] for name in os.listdir(root)
                            if name.endswith(".txt") and achar_arquivo_wildcard(name))
    except OSError as error:
        logging.warning(f"[Impact Wildcard] Error listing custom_wildcards: {error}")
        categories = []
    return categories or ["styles", "poses"]


def entry_with_strength(entry, strength):
    def replace(match):
        name = match.group(1).split(":", 1)[0].strip()
        return f"<lora:{name}:{strength}:{strength}>"
    return _LORA_RE.sub(replace, entry, count=1)


def nome_arquivo_limpo(nome, fallback="no_lora"):
    name = nome.replace("\\", "/").split("/")[-1]
    name = re.sub(r'[<>:"/\\|?*\[\],]', "", name)
    return re.sub(r"\s+", "_", name).strip("_") or fallback


def resolver_caminho_lora(nome):
    """Locates the .safetensors (or other supported formats) in the ComfyUI cache, resolving subfolders."""
    if not nome.endswith((".safetensors", ".ckpt", ".pt")):
        nome_com_ext = nome + ".safetensors"
    else:
        nome_com_ext = nome

    todos_loras = folder_paths.get_filename_list("loras")

    # Attempt 1: Exact match or relative path
    for lora in todos_loras:
        lora_norm = lora.replace("\\", "/")
        nome_norm = nome_com_ext.replace("\\", "/")
        if lora_norm == nome_norm or lora_norm.endswith("/" + nome_norm):
            return lora

    # Attempt 2: Filename only (ignores subfolders)
    nome_basico = os.path.basename(nome_com_ext)
    for lora in todos_loras:
        if os.path.basename(lora) == nome_basico:
            return lora

    return None

def extrair_e_limpar_loras(prompt):
    """Scans the prompt for <lora:NAME:weight> or <lora:NAME:model_w:clip_w> tags,
    returning the list of LoRAs to load and cleaning the tags from the text."""
    if not prompt:
        return [], ""

    pattern = r"<lora:([^>]+)>"
    matches = re.findall(pattern, prompt)

    loras_extraidos = []
    for match in matches:
        parts = match.split(":")
        if len(parts) >= 1:
            nome_lora = parts[0].strip()
            weight_model = 1.0
            weight_clip = 1.0

            if len(parts) >= 2:
                try:
                    weight_model = float(parts[1].strip())
                    weight_clip = weight_model
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    weight_clip = float(parts[2].strip())
                except ValueError:
                    pass

            loras_extraidos.append((nome_lora, weight_model, weight_clip))

    # Clean the tags from the prompt
    prompt_limpo = re.sub(pattern, "", prompt)

    # Remove duplicate/trailing commas and extra spaces
    prompt_limpo = re.sub(r",\s*,", ",", prompt_limpo)
    prompt_limpo = re.sub(r"\s+", " ", prompt_limpo).strip()
    prompt_limpo = prompt_limpo.strip(",").strip()

    return loras_extraidos, prompt_limpo

def aplicar_loras_e_codificar(model, clip, loras_a_aplicar, prompt_limpo):
    """Loads LoRAs sequentially and encodes the prompt using CLIPTextEncode (supporting BREAK)."""
    loader = nodes.LoraLoader()

    # 1. Load LoRAs onto the model and CLIP
    for nome_lora, w_model, w_clip in loras_a_aplicar:
        caminho_resolvido = resolver_caminho_lora(nome_lora)
        if caminho_resolvido:
            logging.info(f"[Impact Wildcard] Applying LoRA: {caminho_resolvido} (model={w_model}, clip={w_clip})")
            model, clip = loader.load_lora(model, clip, caminho_resolvido, w_model, w_clip)
        else:
            logging.warning(f"[Impact Wildcard] WARNING: LoRA '{nome_lora}' could not be found in ComfyUI.")

    # 2. Split by BREAK and encode (for composite conditioning / ImpactWildcardEncode compatibility)
    secoes = [x.strip() for x in prompt_limpo.split("BREAK")]
    secoes = [x for x in secoes if x != ""]
    if not secoes:
        secoes = [""]

    encoder = nodes.CLIPTextEncode()
    result_conditioning = None

    for secao in secoes:
        logging.info(f"[Impact Wildcard] Encoding prompt: '{secao}'")
        cond = encoder.encode(clip, secao)[0]
        if result_conditioning is not None:
            result_conditioning = nodes.ConditioningConcat().concat(result_conditioning, cond)[0]
        else:
            result_conditioning = cond

    return model, clip, result_conditioning


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
            loras, text = extrair_e_limpar_loras(prompt)
            result_model, result_clip, conditioning = aplicar_loras_e_codificar(model, clip, loras, text)
            values = (result_model, result_clip, conditioning, text, nome_arquivo_limpo(name), label)
            for output, value in zip(outputs, values):
                output.append(value)
        return outputs
