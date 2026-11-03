# Test entry-point inventory

Every tracked Python, shell and JavaScript file under `tests` is classified below. Support modules and manual inspection/restart wrappers are not independent tests. Existing failures remain visible in their logs and in [diagnostics.md](diagnostics.md). Repeated runs are not added to the unique pytest total.

| File | Classification / evidence |
|---|---|
| `tests/e2e_animatediff_tutorials.py` | [animatediff-native.log](cpu/animatediff-native.log) |
| `tests/e2e_conditionings.py` | [conditioning.log](gpu/conditioning.log) |
| `tests/e2e_dd_compat.py` | [detailer.log](detailer-final/detailer.log) |
| `tests/e2e_lora_wildcard_inference.py` | [lora-inference.log](inference/lora-inference.log) |
| `tests/e2e_regional.py` | [regional.log](gpu/regional.log) |
| `tests/e2e_tutorials.py` | [switch-tutorials.log](gpu-retry/switch-tutorials.log) |
| `tests/e2e_ui.py` | support-module |
| `tests/e2e_wildcards.py` | [wildcard-ui.log](cpu/wildcard-ui.log) · [wildcard-ui.log](ondemand/wildcard-ui.log) |
| `tests/integration_lora_wildcard_tensors.py` | [lora-tensors.log](lora-tensors.log) |
| `tests/integration_lora_wildcard_validation.py` | [lora-validation.log](lora-validation.log) |
| `tests/integration_wildcard_cache.py` | [merged-integration.json](../merged-integration.json) |
| `tests/integration_wildcard_cache_tensors.py` | [cache-tensors.json](../cache-tensors.json) |
| `tests/integration_wildcard_tutorial_tensors.py` | [tutorial-tensors.json](tutorial-tensors.json) |
| `tests/legacy_sam.test.mjs` | [legacy-sam-js-rerun.log](legacy-sam-js-rerun.log) |
| `tests/lora_source_case.py` | support-module |
| `tests/lora_wildcard_source.test.mjs` | [lora-source-js.log](lora-source-js.log) |
| `tests/restart_test_server.sh` | manual-utility-not-independent-test |
| `tests/test_animatediff_assembly.py` | [animatediff-unit.xml](animatediff-unit.xml) |
| `tests/test_config_quotes.sh` | [test.log](legacy-with-data/test_config_quotes/test.log) |
| `tests/test_deep_nesting.sh` | [test.log](legacy-with-data/test_deep_nesting/test.log) |
| `tests/test_dynamic_prompts_full.sh` | [test.log](legacy-with-data/test_dynamic_prompts_full/test.log) |
| `tests/test_e2e_legacy_sam.py` | [legacy-sam.xml](cpu/legacy-sam.xml) |
| `tests/test_e2e_lora_wildcard_source.py` | [lora-source.xml](lora/lora-source.xml) |
| `tests/test_e2e_mask_to_segs.py` | [mask-native.xml](cpu/mask-native.xml) |
| `tests/test_e2e_mask_tutorials.py` | [mask-native.xml](cpu/mask-native.xml) |
| `tests/test_e2e_sam_masks.py` | [sam.xml](gpu/sam.xml) |
| `tests/test_e2e_sam_tutorials.py` | [sam.xml](gpu/sam.xml) |
| `tests/test_e2e_switch_loading.py` | [switches.xml](../catalog/switches.xml) |
| `tests/test_e2e_switches.py` | [switches.xml](../catalog/switches.xml) |
| `tests/test_e2e_ui.py` | [browser-harness.xml](../browser-harness.xml) |
| `tests/test_e2e_widget_catalog.py` | [catalog.xml](../catalog/catalog.xml) |
| `tests/test_e2e_wildcard_cache.py` | [cache-controls.xml](../classic/cache-controls.xml) · [cache.xml](../verified-classic/cache.xml) · [cache.xml](../verified-lru/cache.xml) |
| `tests/test_e2e_wildcard_controls.py` | [cache-controls.xml](../classic/cache-controls.xml) · [bypass.xml](../baseline-controls/bypass.xml) |
| `tests/test_e2e_wildcard_tutorials.py` | [tutorial-native.xml](../catalog/tutorial-native.xml) · [lbw-parity.xml](lbw-candidate/lbw-parity.xml) |
| `tests/test_edge_cases.sh` | [test.log](legacy-with-data/test_edge_cases/test.log) |
| `tests/test_encoding.sh` | [encoding-api.log](cpu/encoding-api.log) · [encoding-api.log](ondemand/encoding-api.log) |
| `tests/test_error_handling.sh` | [test.log](legacy-with-data/test_error_handling/test.log) |
| `tests/test_mask_blur.py` | [mask-blur.xml](mask-blur.xml) |
| `tests/test_mask_to_segs.py` | [sam-mask-unit.xml](sam-mask-unit.xml) |
| `tests/test_ondemand_loading.sh` | [test.log](legacy-with-data/test_ondemand_loading/test.log) |
| `tests/test_sam_api.py` | [sam-api.xml](gpu/sam-api.xml) |
| `tests/test_sam_masks.py` | [sam-mask-unit.xml](sam-mask-unit.xml) |
| `tests/wildcard_tutorial_cases.py` | support-module |
| `tests/wildcards/conftest.py` | support-module |
| `tests/wildcards/find_deep_transitive.py` | manual-utility-not-independent-test |
| `tests/wildcards/find_transitive_wildcards.sh` | manual-utility-not-independent-test |
| `tests/wildcards/populate_helpers.py` | support-module |
| `tests/wildcards/run_quick_test.sh` | manual-utility-not-independent-test |
| `tests/wildcards/test_config.py` | [merged-unit.xml](../merged-unit.xml) |
| `tests/wildcards/test_lazy_load_api.sh` | [test.log](legacy-with-data/test_lazy_load_api/test.log) |
| `tests/wildcards/test_lazy_load_verification.py` | [test_lazy_load_verification.log](legacy-python-with-data/test_lazy_load_verification.log) · [test_lazy_load_verification.log](legacy-python-baseline/test_lazy_load_verification.log) |
| `tests/wildcards/test_lora_wildcard_source.py` | [merged-unit.xml](../merged-unit.xml) |
| `tests/wildcards/test_populate_regression.py` | [merged-unit.xml](../merged-unit.xml) |
| `tests/wildcards/test_progressive_loading.py` | [test_progressive_loading.log](legacy-python-with-data/test_progressive_loading.log) · [test_progressive_loading.log](legacy-python-baseline/test_progressive_loading.log) |
| `tests/wildcards/test_progressive_ondemand.sh` | [test.log](legacy-with-data/test_progressive_ondemand/test.log) |
| `tests/wildcards/test_seed_regression.py` | [merged-unit.xml](../merged-unit.xml) |
| `tests/wildcards/test_sequential_loading.sh` | [test.log](legacy-with-data/test_sequential_loading/test.log) |
| `tests/wildcards/test_versatile_prompts.sh` | [test.log](legacy-with-data/test_versatile_prompts/test.log) |
| `tests/wildcards/test_wildcard_consistency.sh` | [test.log](legacy-with-data/test_wildcard_consistency/test.log) |
| `tests/wildcards/test_wildcard_encoding_strict.py` | [merged-unit.xml](../merged-unit.xml) |
| `tests/wildcards/test_wildcard_final.py` | [result.json](legacy-final/result.json) · [test.log](legacy-final/test.log) |
| `tests/wildcards/test_wildcard_lazy_loading.py` | [test_wildcard_lazy_loading.log](legacy-python-with-data/test_wildcard_lazy_loading.log) · [test_wildcard_lazy_loading.log](legacy-python-baseline/test_wildcard_lazy_loading.log) |
| `tests/wildcards/verify_ondemand_mode.sh` | manual-utility-not-independent-test |
