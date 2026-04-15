import argparse
import logging
from argparse import Namespace
from pathlib import Path

import torch
from brevitas.export import export_onnx_qcdq
from model.dataset import load_speech_commands, speech_commands_class
from model.net import QuantKWS
from rich.logging import RichHandler
from torch import nn, optim

from tools import quantize
from tools.util import evaluate, train_one_epoch

# current working directory of this script
cwd = Path(__file__.rsplit("/", 1)[0])

# configure logging to use RichHandler for better console output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=True,
            show_time=False,
        )
    ],
)


def train(net: str, save_dir: Path, args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = load_speech_commands(args.dataset, args.batch_size)
    input_t = torch.randn(1, 1, 28, 28).to(device)
    model = QuantKWS(len(speech_commands_class)).to(device)

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            logging.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            logging.info(f"Checkpoint loaded successfully (epoch={checkpoint['epoch']}, best_test_acc={checkpoint['best_test_acc']:.4f})")
        else:
            logging.warning(f"Checkpoint file {ckpt_path} not found. Starting from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare to save the best checkpoint based on test accuracy
    best_ckpt_path = save_dir / f"{net}.pth"
    best_test_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logging.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_acc": best_test_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            logging.info(f"New best model saved to {best_ckpt_path} (test_acc={best_test_acc:.4f})")

    logging.info(f"Training completed. Best test_acc={best_test_acc:.4f}, checkpoint saved at {best_ckpt_path}")
    export_onnx_qcdq(model, input_t=input_t, export_path=save_dir / f"{net}.onnx", opset_version=13)
    logging.info(f"Model exported to ONNX format at {save_dir / f'{net}.onnx'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="/path/to/SpeechCommands")
    parser.add_argument("--save-dir", type=str, default="model/output")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--skip-train", default=False, action="store_true", help="Whether to skip training")
    args = parser.parse_args()

    # get the current working directory of this script
    net = "kws"
    save_dir = Path(cwd / args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train(net, save_dir, args)
    quantize(save_dir / f"{net}.onnx", cwd / f"{net}_params", layout="HWC")


if __name__ == "__main__":
    main()
