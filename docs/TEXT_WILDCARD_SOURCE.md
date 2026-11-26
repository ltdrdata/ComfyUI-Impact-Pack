# TextWildcardSource Node

The `TextWildcardSource` is the plain-text sibling of [`LoraWildcardSource`](LORA_WILDCARD_SOURCE.md). It turns any wildcard file into a list of reusable prompt presets: pick a line from a dropdown, or let the seed walk through the file in order or pick one at random.

It reads the same directory (`custom_wildcards` in `impact-pack.ini`) with the same rules: each top-level `.txt` file is a category, the same saved aliases apply, and files outside that directory cannot be read through the category input or the HTTP route. Unlike `LoraWildcardSource`, every line is selectable, with or without a `<lora:...>` tag, and no model is loaded: the node only outputs text.

## Modes

- **fixed**: Outputs the line selected in the `text_wildcard` dropdown. If that line has been removed from the file, execution reports the missing selection instead of silently choosing another one.
- **sequential**: Outputs line number `seed % line_count`. With the seed's *control after generate* set to `increment`, every queued run moves to the next line and wraps around at the end of the file.
- **random**: Picks a line from the seed, so the same seed always gives the same line.
- **batch**: Outputs every line as a list, one queued item per line.

Wildcard syntax inside the chosen line (such as `__color__` or `{red|blue}`) is processed with the same seed, so a line can still contain variations.

## Inputs

- **category** (COMBO): The text file names (categories) in the configured directory.
- **text_wildcard** (COMBO): The lines of the selected category. Long lines are shortened to 60 characters in the dropdown; repeated labels get a ` (1)`, ` (2)`... suffix. Changing the category refreshes this list from `/impact/get_text_items`.
- **mode** (COMBO): `fixed`, `random`, `sequential` or `batch`.
- **seed** (INT): Seed used for the sequential index, the random choice and wildcard processing inside the line.

## Outputs

- **text** (STRING): The processed line (a list in `batch` mode). An empty category outputs an empty string.
- **name** (STRING): The dropdown label of the line, useful as a filename prefix.

## Example

With `custom_wildcards/scenes.txt`:

```
A neon-lit alley at night, rain-slicked pavement, medium shot, cinematic rim light.
A sunlit Mediterranean street at golden hour, wide shot, warm side light.
A misty forest at dawn, low angle, soft volumetric light.
```

set `category` to `scenes`, `mode` to `sequential` and the seed control to `increment`: three runs render the three scenes in order. Two `TextWildcardSource` nodes joined by `StringConcatenate` give independent preset lists for different parts of a prompt (for example, the scene and the characters).
