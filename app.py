from typing import List, Union
import json
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from models.transceiver import DeepSC
from utils import greedy_decode, SeqtoText, SNR_to_noise


class PredictRequest(BaseModel):
    tokens: Union[List[int], List[List[int]]]
    snr: float = 6.0
    channel: str = 'Rayleigh'


app = FastAPI(title='DeepSC Inference')

# Configuration (can be changed via env vars)
CHECKPOINT = os.environ.get('DEEPSC_CHECKPOINT', 'checkpoints/deepsc-Rayleigh/checkpoint_20.pth')
VOCAB_FILE = os.environ.get('DEEPSC_VOCAB', 'europarl/vocab.json')
NUM_LAYERS = int(os.environ.get('DEEPSC_NUM_LAYERS', 4))
D_MODEL = int(os.environ.get('DEEPSC_D_MODEL', 128))
NUM_HEADS = int(os.environ.get('DEEPSC_NUM_HEADS', 8))
DFF = int(os.environ.get('DEEPSC_DFF', 512))
MAX_DECODE_LEN = int(os.environ.get('DEEPSC_MAX_DECODE_LEN', 30))


@app.on_event('startup')
def load_model():
    global model, stoT, device, pad_idx, start_idx, end_idx

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    vocab = json.load(open(VOCAB_FILE, 'rb'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx['<PAD>']
    start_idx = token_to_idx['<START>']
    end_idx = token_to_idx['<END>']
    num_vocab = len(token_to_idx)

    # instantiate model with same signature as training
    model = DeepSC(NUM_LAYERS, num_vocab, num_vocab, num_vocab, num_vocab,
                   D_MODEL, NUM_HEADS, DFF, 0.1)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    stoT = SeqtoText(token_to_idx, end_idx)


def _pad_batch(token_seqs: List[List[int]], pad_value: int) -> torch.LongTensor:
    max_len = max(len(s) for s in token_seqs)
    padded = [s + [pad_value] * (max_len - len(s)) for s in token_seqs]
    return torch.LongTensor(padded)


def _clean_outputs(batch_out, start_symbol: int):
    cleaned = []
    for seq in batch_out:
        if len(seq) > 0 and seq[0] == start_symbol:
            seq = seq[1:]
        cleaned.append(seq)
    return cleaned


@app.post('/predict')
def predict(req: PredictRequest):
    """Accepts token indices (single sequence or batch) and returns decoded text.

    Expected tokens: either a single list of ints or a list of lists of ints.
    """
    # normalize to batch
    tokens = req.tokens
    if len(tokens) == 0:
        return {'texts': []}

    # detect single sequence vs batch
    if isinstance(tokens[0], int):
        batch_seqs = [tokens]
    else:
        batch_seqs = tokens

    # pad and create tensor
    src = _pad_batch(batch_seqs, pad_idx).to(device)

    noise_std = SNR_to_noise(req.snr)
    with torch.no_grad():
        out = greedy_decode(model, src, noise_std, MAX_DECODE_LEN, pad_idx, start_idx, req.channel)

    out_list = out.cpu().numpy().tolist()
    out_clean = _clean_outputs(out_list, start_idx)
    texts = [stoT.sequence_to_text(seq) for seq in out_clean]

    return {'texts': texts}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='0.0.0.0', port=8000, log_level='info')
