import argparse
import logging
from argparse import Namespace
from pathlib import Path

import onnx
import torch
from brevitas.export import export_onnx_qcdq
from model.dataset import load_fashion_mnist
from model.net import QuantMLP
from rich.logging import RichHandler
from torch import nn, optim

from tools.dump import dump
from tools.fusion import fusion
from tools.parser import onnx_parse
from tools.planner import Planner

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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def train(net: str, save_dir: Path, args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = load_fashion_mnist(args.dataset, args.batch_size)
    input_t = torch.randn(1, 1, 28, 28).to(device)
    model = QuantMLP().to(device)

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


def quantize(net: str, save_dir: Path):
    model = onnx.load(save_dir / f"{net}.onnx")

    # get shape info and initializers from the model, which may be used for fusion and dumping
    shape_map, init_map = onnx_parse(model)

    # perform operator fusion to optimize the model, which may modify the graph structure and initializers
    model = fusion(model, init_map)

    # create a memory planner to analyze tensor lifetimes and compute memory offsets, which may be used for dumping
    planner = Planner(model, shape_map)

    # dump the optimized model and memory plan to files, which may be used for code generation and execution
    dump(model, shape_map, init_map, planner, cwd / f"{net}_params")
    onnx.save(model, save_dir / f"{net}_fused.onnx")  # save the fused model for Netron visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="/path/to/FashionMNIST")
    parser.add_argument("--save-dir", type=str, default="model/output")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # get the current working directory of this script
    net = "mlp"
    save_dir = Path(cwd / args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train(net, save_dir, args)
    quantize(net, save_dir)


if __name__ == "__main__":
    main()
