# ============================================================
# STANDALONE LOGGER
# Independent utility for ComfyUI-Impact-Pack
# ============================================================

# ANSI Escape Codes for console colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Global configuration for the whole pack
PREFIX = "[Impact Core]"
DEBUG = True


def info(msg):
    """Prints a high-level status message in Cyan."""
    print(f"{CYAN}{PREFIX} {msg}{RESET}")


def debug(msg):
    """Prints technical trace messages in standard color (only if DEBUG is True)."""
    if DEBUG:
        print(f"{PREFIX} {msg}")


def warn(msg):
    """Prints a highlighted warning in Yellow."""
    print(f"{YELLOW}{PREFIX} {msg}{RESET}")


def error(msg):
    """Prints a critical error in Red."""
    print(f"{RED}{PREFIX} {msg}{RESET}")
