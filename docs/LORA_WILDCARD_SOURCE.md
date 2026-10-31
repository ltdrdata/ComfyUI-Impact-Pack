# LoraWildcardSource Node

The `LoraWildcardSource` is a generic, dynamic node designed to simplify and automate the application of LoRAs and prompt encoding in ComfyUI, replacing complex text-concatenation setups.

It scans the directory configured by `custom_wildcards` in `impact-pack.ini`, treating each top-level `.txt` file name as a category dropdown option. Selecting a category updates the second dropdown with the corresponding LoRA entries. Files and symlinks outside that directory cannot be read through the category input or HTTP route.

Saved category aliases keep their original fallback: an exact filename wins first; otherwise `style`, `styles`, `estilo` and `estilos_arte` try `styles.txt` then `estilos_arte.txt`, while `pose` and `poses` try `poses.txt` then `pose.txt`. Every candidate must remain inside the configured directory. The dropdown and execution use the same resolution order.

The optional sample is [example_workflows/wildcards/styles.txt](../example_workflows/wildcards/styles.txt). It is outside the automatically scanned directories so installing this node does not replace an existing `__styles__` wildcard. To use it, copy it into your configured directory under an unused filename, then select that category. Do not overwrite your own wildcard files.

Entries recognized by the original node keep their labels and order. Newly supported entries are appended to the dropdown with unused labels and require an explicit `fixed` selection. The automatic modes keep the original candidate list, so saved seeds, sequential wraparound and batch counts are unchanged by syntax expansion. A saved `A` or `A (1)` still selects the same original row. For an empty category, an ordinary base prompt retains its original text and punctuation; LoRA tags in that prompt are still applied and removed before encoding.

## Features

- **Dynamic Category Mapping**: Automatically populates the `category` dropdown by scanning all `.txt` files in the `custom_wildcards/` directory.
- **Dynamic Options Update**: Changing the `category` in the UI dynamically fetches and updates the available LoRAs in the `lora_wildcard` dropdown.
- **LoRA Loading & Encoding**: Applies LoRA tags in order using ComfyUI's LoRA loader, then encodes the cleaned prompt with `BREAK` support. Text and additional LoRAs after the first tag are preserved. Repeated LoRAs are applied separately. Prompt text such as `__color__` or `{red|blue}` remains literal, matching the original node.
- **Cleanup**: Strips `<lora:...>` tags from the final text prompt before encoding.
- **Outputs for File Saving**: Provides `lora_name` (clean filename) and `name` (label) outputs, which can be connected directly to `SaveImage` as a filename prefix.

## Modes

- **fixed**: Applies the selected entry, including negative, dual-weight, scientific-notation or weightless tags. If that entry has been removed while original candidates remain, execution reports the missing selection instead of silently choosing another LoRA. An original empty selection remains a base-prompt fallback when the file contains only extended-only rows. Workflow loading preserves the saved selection.
- **random**: Randomly chooses one LoRA from the active category based on the seed.
- **sequential**: Progressively cycles through the list of LoRAs in the category using the seed (`seed % total_loras`).
- **batch**: Runs every eligible entry in a single queue batch.

The automatic modes (`random`, `sequential`, `batch`) only include rows containing a tag in the original single nonnegative-weight format, such as `<lora:name:0.7>`. They ignore rows recognized only through the extended syntax, even if one is selected in the dropdown. If there are no eligible rows, they encode the base prompt without a category LoRA, as before. Use `fixed` to apply an extended-only row.

## Inputs

- **model** (MODEL): The input model.
- **clip** (CLIP): The input CLIP model.
- **category** (COMBO): Dropdown containing the text file names (categories) from the `custom_wildcards/` directory.
- **lora_wildcard** (COMBO): The LoRA entries inside the selected category file.
- **mode** (COMBO): Select between `fixed`, `random`, `sequential`, or `batch`.
- **strength** (FLOAT): Overrides the model and CLIP weights of the selected entry's primary LoRA (Default: `0.8`). For an original entry, this is the first tag recognized by the original node; for a newly supported entry, it is the first LoRA tag. Other LoRAs retain their declared numeric weights. Block-weight options are not supported by this node.
- **base_prompt** (STRING): Multiline text input for the base prompt.
- **seed** (INT): Seed used for random selection and sequential index. Image sampling uses the sampler's own seed.

## Outputs

- **model** (MODEL): The modified model with the LoRA applied.
- **clip** (CLIP): The modified CLIP with the LoRA applied.
- **conditioning** (CONDITIONING): The encoded positive conditioning.
- **text** (STRING): The cleaned text prompt (without the `<lora:...>` tags).
- **lora_name** (STRING): The clean filename of the applied LoRA (useful as a filename prefix).
- **name** (STRING): The label of the applied LoRA entry.
