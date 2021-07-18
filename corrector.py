import os
from abc import ABC, abstractmethod
from typing import List

import torch
from helpers import load_vocab_dict, get_model_nparams

def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


class Corrector(ABC):
    DEFAULT_CHECKPOINT_PATH = {
        "rubert-base-cased-conversational": 'D:/Google/NeuroNet/BERT/SpellBert/new_models',
    }
    
    # TODO: deprecated usage; should be reoved in next versions
    DEFAULT_CHECKERNAME_TO_NAME_MAPPING = {
        "BertChecker": "rubert-base-cased-conversational",
    }


    def __init__(self, **kwargs):

        self._default_name = kwargs.get("name", None)
        self.tokenize = kwargs.get("tokenize", True)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if self.device == "gpu" else self.device

        self.ckpt_path, self.vocab_path, self.weights_path = None, None, None
        self.model, self.vocab = None, None

        if not self._default_name:

            try:
                self._default_name = Corrector.DEFAULT_CHECKERNAME_TO_NAME_MAPPING[self.__class__.__name__]
            except KeyError as e:
                msg = f"Unable to resolve checker name {self.__class__.__name__} " 
                raise ModuleNotFoundError(msg) from e

        if kwargs.get("pretrained", False):
            self.from_pretrained(ckpt_path=self.ckpt_path)

    def is_model_ready(self):
        assert not (self.model is None or self.vocab is None), print("model & vocab must be loaded first")

    @property
    def get_device(self):
        return self.device

    def set_device(self, device='cpu'):
        prev_device = self.device
        device = "cuda" if ((device == "gpu" or device == "cuda") and torch.cuda.is_available()) else "cpu"
        if not (prev_device == device):

            if self.model is not None:
                try:
                    self.model.to(device)
                except Exception as e:
                    try:
                        self.from_pretrained(self.ckpt_path, vocab=self.vocab_path)
                    except Exception as e:
                        msg = f"Unable to move model from {prev_device} to {device}. " \
                              f"Please load a new instance with argument `device={device}. "
                        raise Exception(msg)
            self.device = device
            print(f"model set to work on {device}")
        return

    def correct(self, x):
        return self.correct_string(x)

    def correct_string(self, mystring: str, return_all=False) -> str:
        x = self.correct_strings([mystring], return_all=return_all)
        if return_all:
            return x[0][0], x[1][0]
        else:
            return x[0]

    def correct_from_file(self, src, dest="./clean_version.txt"):
        self.is_model_ready()
        x = [line.strip() for line in open(src, 'r')]
        y = self.correct_strings(x)
        print(f"saving results at: {dest}")
        opfile = open(dest, 'w')
        for line in y:
            opfile.write(line + "\n")
        opfile.close()
        return

    def _from_pretrained(self, ckpt_path=None, vocab_path=None):

        if ckpt_path:
            self._default_name = os.path.split(ckpt_path)[-1]
            self.ckpt_path = ckpt_path
        else:
            # self._default_name is kept default
            self.ckpt_path = Corrector.DEFAULT_CHECKPOINT_PATH[self._default_name]

        self.vocab_path = vocab_path or os.path.join(self.ckpt_path, "vocab.pkl")
        self.load_output_vocab(self.vocab_path)
        self.load_model(self.ckpt_path)

        return

    def from_pretrained(self, ckpt_path=None, vocab_path=None, **kwargs):
        self._from_pretrained(ckpt_path = ckpt_path, vocab_path = vocab_path, **kwargs)

    def load_output_vocab(self, vocab_path):
        print(f"loading vocab from path:{vocab_path}")
        self.vocab = load_vocab_dict(vocab_path)

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def finetune(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, ckpt_path):
        raise NotImplementedError

    @abstractmethod
    def correct_strings(self, mystrings: List[str], return_all=False):
        raise NotImplementedError

    @property
    def get_num_params(self):
        self.is_model_ready()
        return get_model_nparams(self.model)

    def model_size(self, model=None):
        if not model:
            model = self.model
            self.is_model_ready()
        sz = {
            "num_params": get_model_nparams(model),
            "disk_size (in MB)": get_size_of_model(model),
        }
        return sz

