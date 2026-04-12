import logging

from .patches import install_patches as _install_patches

_LOG = logging.getLogger(__name__)
_PATCHES_INSTALLED = False


def install_patches() -> None:
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _install_patches()
    _PATCHES_INSTALLED = True
    _LOG.info("GPU Resident Loader: startup patches installed")
