# -*- coding: utf-8 -*-
import os
import re
import random
import logging
import folder_paths
import nodes
from aiohttp import web
from server import PromptServer

_LORA_RE = re.compile(r"<lora:([^:>]+):([\d.]+)\s*>")

def achar_arquivo_wildcard(nome_arquivo):
    """Locates the wildcard .txt file in the custom_wildcards directory of the Impact Pack."""
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _CUSTOM_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "custom_wildcards"))
    
    nome_base, ext = os.path.splitext(nome_arquivo)
    names_to_try = [nome_arquivo]
    
    if nome_base in ("style", "styles", "estilo", "estilos_arte"):
        names_to_try.extend(["styles.txt", "estilos_arte.txt"])
    elif nome_base in ("pose", "poses"):
        names_to_try.extend(["poses.txt", "pose.txt"])
        
    for nome in names_to_try:
        caminho = os.path.join(_CUSTOM_DIR, nome)
        if os.path.isfile(caminho):
            return caminho
    return None

def parse_linhas_wildcard(caminho):
    """Reads the wildcard .txt file and extracts a list of [(label, trigger, lora_name)]."""
    entradas = []
    if not caminho or not os.path.isfile(caminho):
        return entradas
    vistos = {}
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            m = _LORA_RE.search(linha)
            if not m:
                continue
            nome = m.group(1).strip()
            gatilho = linha[: m.start()].rstrip().rstrip(",").strip()
            label = nome
            if label in vistos:
                vistos[label] += 1
                label = f"{nome} ({vistos[label]})"
            else:
                vistos[label] = 0
            entradas.append((label, gatilho, nome))
    return entradas

def listar_categorias_wildcard():
    """Lists all available .txt wildcard category names (without extension) from custom_wildcards directory."""
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _CUSTOM_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "custom_wildcards"))
    
    arquivos_encontrados = set()
    if os.path.isdir(_CUSTOM_DIR):
        try:
            for f in os.listdir(_CUSTOM_DIR):
                if f.endswith(".txt") and not f.endswith(".bak_rename") and not f.endswith(".bak_perso"):
                    # Get filename without extension
                    nome_limpo = os.path.splitext(f)[0]
                    arquivos_encontrados.add(nome_limpo)
        except Exception as e:
            logging.warning(f"[Impact Wildcard] Error listing custom_wildcards: {e}")
            
    retorno = sorted(list(arquivos_encontrados))
    return retorno or ["styles", "poses"]

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

def nome_arquivo_limpo(nome, fallback="no_lora"):
    """Cleans and formats the LoRA filename for safe usage in image saving prefixes."""
    n = nome.replace("\\", "/").split("/")[-1]
    n = re.sub(r'[<>:"/\\|?*\[\],]', "", n)
    n = re.sub(r"\s+", "_", n).strip("_")
    return n or fallback

def fmt_forca(v):
    """Formats LoRA strength as a short, clean readable string."""
    return f"{v:.2f}".rstrip("0").rstrip(".")

# HTTP API route to sync dependent dropdowns in the ComfyUI frontend
@PromptServer.instance.routes.get("/impact/get_items")
async def get_items_handler(request):
    category = request.query.get("category", request.query.get("categoria", ""))
    
    # Map old categories for backward compatibility
    if category == "estilo":
        nome_arquivo = "estilos_arte.txt"
    elif category == "pose":
        nome_arquivo = "poses.txt"
    else:
        nome_arquivo = f"{category}.txt"
        
    caminho = achar_arquivo_wildcard(nome_arquivo)
    entradas = parse_linhas_wildcard(caminho)
    labels = [e[0] for e in entradas]
    return web.json_response(labels)

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
        nome_arquivo = f"{category}.txt"
        caminho = achar_arquivo_wildcard(nome_arquivo)
        entradas = parse_linhas_wildcard(caminho)

        if not entradas:
            m_res, c_res, cond = aplicar_loras_e_codificar(model, clip, [], base_prompt)
            return ([m_res], [c_res], [cond], [base_prompt], ["no_lora"], [f"no_{category}"])

        if mode == "batch":
            escolhidas = entradas
        elif mode == "random":
            escolhidas = [random.Random(seed).choice(entradas)]
        elif mode == "sequential":
            escolhidas = [entradas[seed % len(entradas)]]
        else:  # fixed
            sel = next((e for e in entradas if e[0] == lora_wildcard), entradas[0])
            escolhidas = [sel]

        out_models, out_clips, out_conds = [], [], []
        out_texts, out_lora_names, out_names = [], [], []

        for label, trigger, nome in escolhidas:
            partes = []
            if base_prompt and base_prompt.strip():
                partes.append(base_prompt.strip().rstrip(","))
            if trigger:
                partes.append(trigger)
            
            prompt_completo = ", ".join(partes)
            tag_lora = f"<lora:{nome}:{strength}>"
            prompt_completo = (prompt_completo + " " + tag_lora).strip() if prompt_completo else tag_lora

            loras_a_carregar, prompt_limpo = extrair_e_limpar_loras(prompt_completo)

            m_res, c_res, cond_res = aplicar_loras_e_codificar(model, clip, loras_a_carregar, prompt_limpo)

            out_models.append(m_res)
            out_clips.append(c_res)
            out_conds.append(cond_res)
            out_texts.append(prompt_limpo)
            out_lora_names.append(nome_arquivo_limpo(nome))
            out_names.append(label)

        return (out_models, out_clips, out_conds, out_texts, out_lora_names, out_names)
