import argparse
import io
import json
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import parse_qs

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from hypy_utils.logging_utils import setup_logger
from starlette.middleware.cors import CORSMiddleware
from torch import no_grad, LongTensor

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence


log = setup_logger()

app = FastAPI()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

language_marks = {
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_text(text: str, is_symbol: bool):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def tts_fn(text: str, speaker: str, language: str, speed: float):
    if language is not None:
        text = language_marks[language] + text + language_marks[language]
    speaker_id = speaker_ids[speaker]
    stn_tst = get_text(text, False)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return audio


@app.get("/tts/options")
async def get_options():
    return {"speakers": list(speaker_ids.keys()), "languages": list(language_marks.keys())}


@app.post("/tts")
async def generate(request: Request):
    body = (await request.body()).decode()

    # Try parse json
    if body.startswith('{'):
        try:
            data = json.loads(body)
        except JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
    # Try parse x-www-form-urlencoded
    else:
        data = parse_qs(body)
        data = {k: v[0] for k, v in data.items()}

    log.info(data)

    text = data.get('text').strip()
    speaker = data.get('speaker')
    language = data.get('language', '日本語')
    speed = data.get('speed', 1.0)

    if not text or not speaker or language not in language_marks:
        raise HTTPException(status_code=400, detail="Invalid speaker or language (please check /tts/options)")

    audio = tts_fn(text, speaker, language, speed)
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, hps.data.sampling_rate, format='OGG')
    audio_io.seek(0)

    return StreamingResponse(audio_io, media_type='audio/ogg',
                             headers={'Content-Disposition': 'attachment; filename="output.ogg"'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="./OUTPUT_MODEL",
                        help="directory to your fine-tuned model (contains G_latest.pth and config.json)")
    args = parser.parse_args()
    d_config = Path(args.d) / "config.json"
    d_model = Path(args.d) / "G_latest.pth"
    hps = utils.get_hparams_from_file(d_config)

    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = model.eval()

    utils.load_checkpoint(d_model, model, None)
    speaker_ids = hps.speakers

    uvicorn.run(app, host='0.0.0.0', port=27519)
