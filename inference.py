# inference.py
import json
import torch
import argparse
from models.transceiver import DeepSC
from utils import greedy_decode, SeqtoText, SNR_to_noise
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/deepsc-Rayleigh/checkpoint_20.pth', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--max-length', default=30, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--dff', default=512, type=int)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vocab and indices
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = {v:k for k,v in token_to_idx.items()}
pad_idx = token_to_idx['<PAD>']
start_idx = token_to_idx['<START>']
end_idx = token_to_idx['<END>']
num_vocab = len(token_to_idx)

# Build model and load checkpoint
# Use vocabulary size as max length to match the model used during training
model = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab,
               num_vocab, args.d_model, args.num_heads, args.dff, 0.1)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.to(device)
model.eval()

# Helper converter
stoT = SeqtoText(token_to_idx, end_idx)

def infer_sentence(token_indices, noise_snr=6):
    """
    token_indices: torch.LongTensor shape (batch, seq_len)
    noise_snr: SNR value used to compute noise_std via SNR_to_noise
    """
    token_indices = token_indices.to(device)
    noise_std = SNR_to_noise(noise_snr)
    with torch.no_grad():
        out = greedy_decode(model, token_indices, noise_std, args.max_length, pad_idx,
                            start_idx, args.channel)
    out = out.cpu().numpy().tolist()
    texts = [stoT.sequence_to_text(seq) for seq in out]
    return texts

# Example usage with dataset:
if __name__ == '__main__':
    ds = EurDataset('test')  # or build a dataset from raw strings if you have tokenizer
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_data)
    for batch in dl:
        preds = infer_sentence(batch, noise_snr=6)
        print(preds)
        break