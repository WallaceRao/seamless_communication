# refer to https://github.com/facebookresearch/seamless_communication/blob/main/Seamless_Tutorial.ipynb

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import io
import json
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import mmap
import numpy
import soundfile
import torchaudio
import torch

#from collections import defaultdict
#from IPython.display import Audio, display
#from pathlib import Path
from pydub import AudioSegment
ssl._create_default_https_context = ssl._create_unverified_context
from seamless_communication.inference import Translator
ssl._create_default_https_context = ssl._create_unverified_context
#from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover

# Initialize a Translator object with a multitask model, vocoder on the GPU.

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)

tgt_langs = ("spa", "fra", "deu", "ita", "hin", "cmn")

for tgt_lang in tgt_langs:
    text_output, speech_output = translator.predict(
        input="Hey everyone! I hope you're all doing well. Thank you for attending our workshop.",
        task_str="t2st",
        tgt_lang=tgt_lang,
        src_lang="eng",
    )
    print(f"Translated text in {tgt_lang}: {text_output[0]}")
    print()





