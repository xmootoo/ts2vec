import torch
from datautils import load_forecast_csv
from finetune import fine_tune
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune TS2vec model on downstream task')

parser.add_argument('--name', type=str, default="ETTh1", help='Name of dataset')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--seq_len', type=int, default=336, help='Sequence length')
parser.add_argument('--pred_lens', type=int, nargs='+', default=[96], help='Prediction lengths')

# Parse arguments
args = parser.parse_args()

# Load data and pretrained model

data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(args.name)

print(data.shape)
pretrained_model = torch.load(f"pretrained/{args.name}.pt", map_location=torch.device('cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Count number of parameters
n_params = sum(p.numel() for p in pretrained_model.parameters())
print(f"TS2vec Model with {n_params:,} parameters loaded")

# Run fine-tuning
results = fine_tune(pretrained_model,
              data,
              args.name,
              epochs=args.epochs,
              batch_size=args.batch_size,
              train_slice=train_slice,
              valid_slice=valid_slice,
              test_slice=test_slice,
              seq_len=args.seq_len,
              pred_lens=[96],
              device=device)

# Save logs
with open(f"downstream/{args.name}_{args.seq_len}.log", "w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")
