from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import multiprocessing
import threading
import os
import json
import base64

from datetime import datetime
import logging

import torch
import numpy as np
import re
import time
import uuid

import ssl
import io
import json
import mmap
import soundfile
import torchaudio
import base64
import random
from seamless_communication.inference import Translator
import langs
ssl._create_default_https_context = ssl._create_unverified_context


tts_logger = logging.getLogger("seamless")
tts_logger.setLevel(logging.INFO)
log_path = "/seamless/seamless.log"
handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
tts_logger.addHandler(handler)

temp_data_path = "/seamless/data"

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)

MAX_INPUT_AUDIO_LENGTH = 60
AUDIO_SAMPLE_RATE = 16000

def get_random_file_name():
    now = datetime.now()
    dt_string = now.strftime("%Y%d%m_%H%M%S")
    return dt_string + "_" + str(random.randint(1, 1000))

def preprocess_audio(input_audio: str) -> None:
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        tts_logger.info(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
                    
class Handler(SimpleHTTPRequestHandler):
    def send_post_response(self, response_str):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(response_str.encode("UTF-8"))))
        self.end_headers()
        self.wfile.write(response_str.encode("UTF-8"))
    
    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        req_id = str(uuid.uuid4().fields[-1])[:5]
        tts_logger.info(f"received a request, req_id:{req_id}")
        json_obj = json.loads(data_string)
        err_msg = ""
        input_audio = None
        input_text = None
        speech_output = None
        text_output = None
        input_audio_format = "wav"
        output_audio_format = "wav"
        task = ""
        input_lang = "English"
        output_lang = "Mandarin Chinese"
        if "input_audio" in json_obj.keys() and len(json_obj["input_audio"]) > 0:
            input_audio = json_obj["input_audio"]
        if "input_audio_format" in json_obj.keys() and len(json_obj["input_audio_format"]) > 0:
            input_audio_format = json_obj["input_audio_format"]
        if "output_audio_format" in json_obj.keys() and len(json_obj["output_audio_format"]) > 0:
            output_audio_format = json_obj["output_audio_format"]
        if "input_text" in json_obj.keys():
            input_text = json_obj["input_text"]
        if "task" in json_obj.keys() and len(json_obj["task"]) > 0:
            task = json_obj["task"]
        if "input_lang" in json_obj.keys() and len(json_obj["input_lang"]) > 0:
            input_lang = json_obj["input_lang"]
        if "output_lang" in json_obj.keys() and len(json_obj["output_lang"]) > 0:
            output_lang = json_obj["output_lang"]
        if input_lang not in langs.LANGUAGE_NAME_TO_CODE.keys() or \
            output_lang not in langs.LANGUAGE_NAME_TO_CODE.keys():
            err_msg = "received bad input language:" + input_lang + "or output language:" + output_lang
            response_str = json.dumps({"err_msg": err_msg})
            self.send_post_response(response_str)
            return
        input_lang_code = langs.LANGUAGE_NAME_TO_CODE[input_lang]
        output_lang_code = langs.LANGUAGE_NAME_TO_CODE[output_lang]
        text_output = None
        speech_output = None
        audio_file_path = None
        if input_audio:
            audio_binary = base64.b64decode(input_audio)
            audio_file_path = temp_data_path + "/" + get_random_file_name() + ".wav"
            with open(audio_file_path, 'wb+') as file:
                file.write(audio_binary)
            preprocess_audio(audio_file_path)
        print("task:", task)
        if task == "T2T":
            if input_text is None:
                err_msg = "no text provided for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            text_output, speech_output = translator.predict(
                input=input_text,
                task_str="T2TT",
                tgt_lang=output_lang_code,
                src_lang=input_lang_code,
            ) 
        elif task == "T2S":
            if input_text is None:
                err_msg = "no text provided for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            text_output, speech_output = translator.predict(
                input=input_text,
                task_str="T2ST",
                tgt_lang=output_lang_code,
                src_lang=input_lang_code,
            )
        elif task == "S2T":
            if audio_file_path is None:
                err_msg = "no audio provided for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            text_output, speech_output = translator.predict(
                input=audio_file_path,
                task_str="S2TT",
                tgt_lang=output_lang_code,
                src_lang=input_lang_code,
            )
        elif task == "S2S":
            if audio_file_path is None:
                err_msg = "no audio provided for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            text_output, speech_output = translator.predict(
                input=audio_file_path,
                task_str="S2ST",
                tgt_lang=output_lang_code,
                src_lang=input_lang_code,
            )
        elif task == "ASR":
            if audio_file_path is None:
                err_msg = "no audio provided for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            if output_lang_code != input_lang_code:
                err_msg = "input and output language should be same for task:" + task
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            text_output, speech_output = translator.predict(
                input=audio_file_path,
                task_str="ASR",
                tgt_lang=output_lang_code,
                src_lang=input_lang_code,
            )
        else:
            err_msg = "received invalid task:" + task
            response_str = json.dumps({"err_msg": err_msg})
            self.send_post_response(response_str)
            return

        out_text = ""
        if text_output:
            out_text = str(text_output[0])
        base64_str = ""
        print("speech_output:", speech_output)
        if speech_output:
            out_wav = speech_output.audio_wavs[0].cpu().detach().numpy()
            out_wav = out_wav * 32768
            out_wav = out_wav.astype(np.int16)
            print("out_wav:", out_wav)
            pcm_bytes = out_wav.tobytes()
            enc_data = pcm_bytes
            base64_str = base64.b64encode(enc_data).decode()
        err_msg = "OK"
        response_str = json.dumps({"sample_rate": "16000", "output_text": out_text,
                                   "output_audio_format":output_audio_format,
                                   "output_audio": base64_str, "err_msg":err_msg})
        self.send_post_response(response_str)
        return

def run():
    global translator
    server = ThreadingHTTPServer(('0.0.0.0', 82), Handler)
    server.serve_forever()

if __name__ == '__main__':
    os.environ["WORK_DIR"] = "./"
    run()