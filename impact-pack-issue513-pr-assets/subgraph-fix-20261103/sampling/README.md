# Promoted seed: three actual Chrome queues per revision

This supersedes the original direct-node screenshots for the current promoted-seed implementation. Previous-PR baseline b1ee82d is the prior implementation rebased onto Main 56a34b9; it still advances promoted fixed seeds. Candidate source SHA-256 is recorded in comparison.json and mapped to the final PR commit in ../commit.json. Exact source snapshots are in sources/.

The native workflow was loaded and converted into a subgraph through Chrome menus, with seed and mode promoted. Queue three follows native Export/reload. Six original Chrome captures have distinct prompt IDs and execution timestamps. Source image and history hashes, cache events, widget values, twelve output RGB hashes and cross-revision parity are checked by overlays/render.py. Original captures are unchanged; SVG/HTML overlays add numbered green/red/orange borders and English notes. Links in the PR remain outside raster images.

Candidate wildcard seed stays 513 for all queues; wildcard, sampler, decode and output are cached on queues two and three. Previous-PR submissions advance 513/514/515 and resample. The independent raw-model sampler advances 9001/9002/9003 in both revisions, has three distinct outputs, and each corresponding baseline/candidate RGB image matches exactly. Initial pancake images also match exactly.

Local installed SD1.5 epicrealism + foodphoto, 512x512, deterministic CUDA/PyTorch attention, no downloads.
