from runlib import *
pytest('legacy-progressive','tests/wildcards/test_progressive_loading.py')
pytest('legacy-lazy','tests/wildcards/test_wildcard_lazy_loading.py')
pytest('legacy-lazy-verification','tests/wildcards/test_lazy_load_verification.py')
