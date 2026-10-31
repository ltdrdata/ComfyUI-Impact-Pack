# LoraWildcardSource Node

The `LoraWildcardSource` is a generic, dynamic node designed to simplify and automate the application of LoRAs and prompt encoding in ComfyUI, replacing complex text-concatenation setups.

It scans the directory configured by `custom_wildcards` in `impact-pack.ini`, treating each top-level `.txt` file name as a category dropdown option. Selecting a category updates the second dropdown with the corresponding LoRA entries. Files and symlinks outside that directory cannot be read through the category input or HTTP route.

The optional sample is [example_workflows/wildcards/styles.txt](../example_workflows/wildcards/styles.txt). It is outside the automatically scanned directories so installing this node does not replace an existing `__styles__` wildcard. To use it, copy it into your configured directory under an unused filename, then select that category. Do not overwrite your own wildcard files.

## Features

- **Dynamic Category Mapping**: Automatically populates the `category` dropdown by scanning all `.txt` files in the `custom_wildcards/` directory.
- **Dynamic Options Update**: Changing the `category` in the UI dynamically fetches and updates the available LoRAs in the `lora_wildcard` dropdown.
- **LoRA Loading & Encoding**: Applies LoRA tags in order using ComfyUI's LoRA loader, then encodes the cleaned prompt with `BREAK` support. Text and additional LoRAs after the first tag are preserved. Repeated LoRAs are applied separately. Prompt text such as `__color__` or `{red|blue}` remains literal, matching the original node.
- **Cleanup**: Strips `<lora:...>` tags from the final text prompt before encoding.
- **Outputs for File Saving**: Provides `lora_name` (clean filename) and `name` (label) outputs, which can be connected directly to `SaveImage` as a filename prefix.

## Modes

- **fixed**: Applies the selected entry. If that entry has been removed, execution reports the missing selection instead of silently choosing another LoRA. Workflow loading preserves the saved selection.
- **random**: Randomly chooses one LoRA from the active category based on the seed.
- **sequential**: Progressively cycles through the list of LoRAs in the category using the seed (`seed % total_loras`).
- **batch**: Runs all LoRAs in the category in a single queue batch.

## Inputs

- **model** (MODEL): The input model.
- **clip** (CLIP): The input CLIP model.
- **category** (COMBO): Dropdown containing the text file names (categories) from the `custom_wildcards/` directory.
- **lora_wildcard** (COMBO): The LoRA entries inside the selected category file.
- **mode** (COMBO): Select between `fixed`, `random`, `sequential`, or `batch`.
- **strength** (FLOAT): Overrides the model and CLIP weights of the entry's first LoRA (Default: `0.8`). Other LoRAs in the entry or base prompt retain their declared numeric weights. Block-weight options are not supported by this node.
- **base_prompt** (STRING): Multiline text input for the base prompt.
- **seed** (INT): Seed used for random selection and sequential index. Image sampling uses the sampler's own seed.

## Outputs

- **model** (MODEL): The modified model with the LoRA applied.
- **clip** (CLIP): The modified CLIP with the LoRA applied.
- **conditioning** (CONDITIONING): The encoded positive conditioning.
- **text** (STRING): The cleaned text prompt (without the `<lora:...>` tags).
- **lora_name** (STRING): The clean filename of the applied LoRA (useful as a filename prefix).
- **name** (STRING): The label of the applied LoRA entry.
