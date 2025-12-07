# inference.py
import json
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy
import torch
import argparse
from models.transceiver import DeepSC
from utils import greedy_decode, SeqtoText, SNR_to_noise
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/deepsc-Rayleigh/checkpoint_20.pth', type=str, help='Path to model checkpoint')
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str, help='Path to vocabulary file')
parser.add_argument('--max-length', default=30, type=int, help='Maximum decoding length')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for inference')
parser.add_argument('--channel', default='Rayleigh', type=str, help='Channel type (e.g., AWGN, Rayleigh, Rician)')
parser.add_argument('--num-layers', default=4, type=int, help='Number of transformer layers')
parser.add_argument('--d-model', default=128, type=int, help='Dimension of the model')
parser.add_argument('--num-heads', default=8, type=int, help='Number of attention heads')
parser.add_argument('--dff', default=512, type=int, help='Dimension of the feed-forward network')
parser.add_argument('--snr', default=10, type=float, help='SNR value used to compute noise standard deviation')
args = parser.parse_args()

if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Using device:", device)

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
    print(f"Using SNR {noise_snr} -> noise: {noise_std}")
    with torch.no_grad():
        out, enc_output, channel_enc_output, Tx_sig, Rx_sig, memory = greedy_decode(model, token_indices, noise_std, args.max_length, pad_idx,
                            start_idx, args.channel)
    out_cpu = out.cpu().numpy().tolist() # Copy to CPU memory, convert to list

    # Convert to text
    texts = [stoT.sequence_to_text(seq) for seq in out_cpu]
    intermediates = {
        "out": out,
        "enc_output": enc_output,
        "channel_enc_output": channel_enc_output,
        "Tx_sig": Tx_sig,
        "Rx_sig": Rx_sig,
        "memory": memory
    }
    return texts, intermediates

# Example usage with dataset:
if __name__ == '__main__':
    ds = EurDataset('test')  # or build a dataset from raw strings if you have tokenizer
    # print("ds.data: ", ds.data[0])
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_data)
    inp_ = dl.dataset.data[0]
    inp_str = stoT.sequence_to_text(inp_)
    print("Input sentence: ", inp_str)
    for batch in dl:
        preds, intermediates = infer_sentence(batch, noise_snr=args.snr)
        for p in preds:
            print(p)
        out = intermediates['out']
        enc_output = intermediates['enc_output']
        channel_enc_output = intermediates['channel_enc_output']
        Tx_sig = intermediates['Tx_sig']
        Rx_sig = intermediates['Rx_sig']
        memory = intermediates['memory']

        tx: numpy.ndarray = Tx_sig.cpu().numpy()[0]
        rx: numpy.ndarray = Rx_sig.cpu().numpy()[0]

        
        # Visualize input and output tensors
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
        fig: Figure = fig
        # Add top-level title for the entire figure
        fig.suptitle(f'DeepSC Inference - SNR: {args.snr} dB, Channel: {args.channel}', fontsize=14, fontweight='bold')
        
        inp = batch.cpu().numpy()[0]
        outp = out.cpu().numpy()[0]
        ax1.plot(inp, marker='o')
        ax1.plot(outp, marker='x')
        ax1.set_title('Input vs Output')
        ax1.grid(True)
        ax1.legend(['Input', 'Output'])
        fig.subplots_adjust(bottom=0.15)
        # Figure fraction (top-left of full figure)
        text = f"Input: {inp_str}\nOutput: {preds[0]}"
        fig.text(0.1, 0.08, text, wrap=True, ha='left', va='top',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.9))
        
        io_diff = []
        for i, o in zip(inp, outp):
            io_diff.append(i - o)
        ax2.plot(tx.flatten(), label='Tx', alpha=0.5)
        ax2.plot(rx.flatten(), label='Rx', alpha=0.5)
        ax2.set_title('Transmitted vs Received Signal')
        ax2.grid(True)
        ax2.legend()
        # ax2.plot(io_diff)
        # ax2.set_title('Input - Output Difference')
        # ax2.axis('off')


        ax3.imshow(tx)
        ax3.set_title('Transmitted Signal')
        ax3.axis('off')

        ax4.imshow(rx)
        ax4.set_title('Received Signal')
        ax4.axis('off')

        plt.show()
        break