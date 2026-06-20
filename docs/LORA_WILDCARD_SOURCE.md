# LoraWildcardSource Node

The `LoraWildcardSource` is a generic, dynamic node designed to simplify and automate the application of LoRAs and prompt encoding in ComfyUI, replacing complex text-concatenation setups.

It scans files inside the `custom_wildcards/` directory, treating each `.txt` file name as a category dropdown option. Selecting a category dynamically updates the second dropdown with the corresponding LoRA wildcards parsed from the file.

## Features

- **Dynamic Category Mapping**: Automatically populates the `category` dropdown by scanning all `.txt` files in the `custom_wildcards/` directory.
- **Dynamic Options Update**: Changing the `category` in the UI dynamically fetches and updates the available LoRAs in the `lora_wildcard` dropdown.
- **Built-in LoRA Loading & Encoding**: Sequentially loads the selected LoRAs onto the model and CLIP and encodes the prompt using `CLIPTextEncode` (supporting the `BREAK` keyword) to output `MODEL`, `CLIP`, and `CONDITIONING` directly.
- **Cleanup**: Strips `<lora:...>` tags from the final text prompt before encoding.
- **Outputs for File Saving**: Provides `lora_name` (clean filename) and `name` (label) outputs, which can be connected directly to `SaveImage` as a filename prefix.

## Modes

- **fixed**: Applies the single selected LoRA from the dropdown.
- **random**: Randomly chooses one LoRA from the active category based on the seed.
- **sequential**: Progressively cycles through the list of LoRAs in the category using the seed (`seed % total_loras`).
- **batch**: Runs all LoRAs in the category in a single queue batch.

## Inputs

- **model** (MODEL): The input model.
- **clip** (CLIP): The input CLIP model.
- **category** (COMBO): Dropdown containing the text file names (categories) from the `custom_wildcards/` directory.
- **lora_wildcard** (COMBO): The LoRA entries inside the selected category file.
- **mode** (COMBO): Select between `fixed`, `random`, `sequential`, or `batch`.
- **strength** (FLOAT): The model and clip strength of the applied LoRA (Default: `0.8`).
- **base_prompt** (STRING): Multiline text input for the base prompt.
- **seed** (INT): Seed used for random selection, sequential index, or generation.

## Outputs

- **model** (MODEL): The modified model with the LoRA applied.
- **clip** (CLIP): The modified CLIP with the LoRA applied.
- **conditioning** (CONDITIONING): The encoded positive conditioning.
- **text** (STRING): The cleaned text prompt (without the `<lora:...>` tags).
- **lora_name** (STRING): The clean filename of the applied LoRA (useful as a filename prefix).
- **name** (STRING): The label of the applied LoRA entry.
