import requests
import numpy as np
import json
import base64
import time
import datetime
import sys    
import http.client

import gradio as gr
import os
import wave
import langs

SUPPORTED_TASKS = {
    "Text-Text":"T2T",
    "Text-Speech":"T2S",
    "Speech-Text":"S2T",
    "Speech-Speech":"S2S",
    "ASR":"ASR"
}

def transfer(wav_file, input_text, task, input_lang, output_lang):
    conn = http.client.HTTPConnection('127.0.0.1:82')
    data_format = "wav" # only MP3 and pcm are supported 
    request_headers = {'Content-type': 'application/json'}
    audio_bytes = None
    base64_str = ""
    if wav_file:
        with open(wav_file, "rb") as file:
            audio_bytes = file.read()
            base64_str = base64.b64encode(audio_bytes).decode()
    foo = {
        'input_audio': base64_str,
        'input_audio_format':data_format,
        'input_text': input_text,
        'task': SUPPORTED_TASKS[task],
        'input_lang': input_lang,
        'output_lang': output_lang,
        'output_audio_format':'wav',
    }
    json_data = json.dumps(foo)
    start = time.time()
    res = conn.request("POST", "", json_data, request_headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    response_bytes = response.read()
    response_headers = response.getheaders()
    json_obj = json.loads(response_bytes)
    err_msg = ""
    output_wav_file = None
    output_text = None
    if "err_msg" in json_obj.keys():
        err_msg = json_obj["err_msg"]
        if err_msg != "OK":
            print("got err msg:", err_msg)
            return None, err_msg
    if "output_audio" in json_obj.keys():
        data_str = json_obj["output_audio"]
        decoded_binary = base64.b64decode(data_str)
        pcm_bytes = decoded_binary
        output_wav_file = "/seamless/data/output.wav"
        with wave.open(output_wav_file, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)
            out_f.setframerate(16000)
            out_f.writeframesraw(pcm_bytes)
    if "output_text" in json_obj.keys():
        output_text = json_obj["output_text"]
    return [output_wav_file, output_text]
        
gr.Markdown(
    """
    # Multi-language Voice Transfer   
    This demo provides an Voice transfer based on seamless communication
    """
)

def process(wav_file, input_text, task, input_lang, output_lang):
    audio_file, text = transfer(wav_file, input_text, task, input_lang, output_lang)
    return audio_file, text
 
demo_inputs = [
    gr.Audio(
        sources=["upload"],
        label="Audio File",
        type="filepath",
    ),
    gr.Textbox(
        label="Input text",
        type="text",
        placeholder="Type something here.."
    ),
    gr.Radio(
        choices=list(SUPPORTED_TASKS.keys()),
        label="Task",
        value="Speech-Speech"
    ),
    gr.Radio(
        choices=list(langs.LANGUAGE_NAME_TO_CODE.keys()),
        label="Input Language",
        value="Speech-Speech"
    ),
    gr.Radio(
        choices=list(langs.LANGUAGE_NAME_TO_CODE.keys()),
        label="Output Lauguage",
        value="Speech-Speech"
    ),
]

demo_outputs = [
    gr.Audio(label="Output Wav"),
    gr.Textbox(
        label="Output text",
        type="text",
    )
]

demo = gr.Interface(
    fn=process,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Seamless Communication Demo",
)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=80)


