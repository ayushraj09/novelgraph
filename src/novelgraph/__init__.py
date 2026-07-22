"""
NovelGraph - typed knowledge-graph pipeline for surfacing untested
Method/Dataset research directions and verifying evidence-backed
hypotheses about them.

See README.md for the pipeline stages and module map.
"""

# Resolves SYSTEM_ROOT_DIRECTORY / DATA_ROOT_DIRECTORY to absolute paths
# anchored to this project, before any submodule's `import cognee` runs.
from . import config  # noqa: F401

__version__ = "0.1.0"
