"""The base error of ``andamentum.proofread``."""


class ProofreadError(Exception):
    """Base class of every error this package raises on purpose."""


class DeckReadError(ProofreadError):
    """A ``.pptx`` file could not be read as a deck."""
