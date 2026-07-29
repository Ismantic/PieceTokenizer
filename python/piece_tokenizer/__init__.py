from pathlib import Path

from ._core import PreTokenizer
from ._core import Tokenizer as _CoreTokenizer


_MODEL_DIR = Path(__file__).resolve().parent / "models"
_PRESETS = {
    "bertc": ("BERTc-Tokenizer.pt", None),
    "summer": ("Summer-Tokenizer.pt", "Summer-Tokenizer.dict.txt"),
}


class Tokenizer(_CoreTokenizer):
    """Tokenizer with optional bundled BERTc and Summer presets."""

    def __init__(self, model=None, *, dict_path=None):
        super().__init__()
        if model is not None:
            self.load(model, dict=dict_path)

    def load(self, model_file, dict=""):
        """Load a bundled preset name or a model path."""
        preset = _PRESETS.get(str(model_file).lower())
        if preset is not None:
            model_name, default_dict_name = preset
            model_file = _MODEL_DIR / model_name
            if not dict and default_dict_name is not None:
                dict = _MODEL_DIR / default_dict_name

        model_path = str(Path(model_file))
        dictionary = "" if not dict else str(Path(dict))
        return super().load(model_path, dictionary)


def BERTcTokenizer():
    """Return a tokenizer loaded with the bundled BERTc vocabulary."""
    return Tokenizer("BERTc")


def SummerTokenizer():
    """Return a tokenizer loaded with the bundled Summer model and dictionary."""
    return Tokenizer("Summer")


def model_path(name):
    """Return the filesystem path of a bundled tokenizer model."""
    preset = _PRESETS.get(str(name).lower())
    if preset is None:
        raise ValueError("name must be 'BERTc' or 'Summer'")
    return _MODEL_DIR / preset[0]


__all__ = [
    "BERTcTokenizer",
    "PreTokenizer",
    "SummerTokenizer",
    "Tokenizer",
    "model_path",
]
