"""Shared inputs and independent expected text for the real-model checks."""

CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"
LORAS = ["SD1.5/etc/add_detail", "SD1.5/etc/detailmaker"]
BASE = "a {red|blue} flower, __color__, <lora: SD1.5/etc/add_detail : +1e-1 : 2e-1> BREAK garden"
FIRST_SECTION = "a {red|blue} flower, __color__,"
LAST_SECTIONS = ["garden, petals, dew", "garden, leaves, mist"]
SAVED_LABEL = LORAS[0] + " (1)"
SAVED_SECTION = "garden, saved second"
FIRST_LABEL_SECTION = "garden,, saved first"
