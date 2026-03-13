"""Lite-Rec CLI entry point."""
from __future__ import annotations

import argparse

from literec import Dataset, BPR, LightGCN, NGCF, Trainer

MODEL_MAP = {"bpr": BPR, "lightgcn": LightGCN, "ngcf": NGCF}


def main():
    parser = argparse.ArgumentParser(description="Lite-Rec: lightweight recommendation")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys())

    # Dataset params
    parser.add_argument("--user_col", default="userId")
    parser.add_argument("--item_col", default="movieId")
    parser.add_argument("--rating_col", default="rating")
    parser.add_argument("--time_col", default="timestamp")
    parser.add_argument("--min_rating", type=float, default=0.0)
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--split", default="loo", choices=["loo", "random"])

    # Model params
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reg_weight", type=float, default=1e-5)

    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 20])

    args = parser.parse_args()

    dataset = Dataset(
        args.data,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        time_col=args.time_col,
        min_rating=args.min_rating,
        min_interactions=args.min_interactions,
        split=args.split,
    )

    model_cls = MODEL_MAP[args.model]
    model_kwargs = {"emb_size": args.emb_size, "reg_weight": args.reg_weight}
    if args.model in ("lightgcn", "ngcf"):
        model_kwargs["n_layers"] = args.n_layers
    if args.model == "ngcf":
        model_kwargs["dropout"] = args.dropout
    model = model_cls(dataset, **model_kwargs)

    trainer = Trainer(
        model, dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        device=args.device,
        seed=args.seed,
        topk=args.topk,
    )
    result = trainer.fit()
    print("\nFinal Test Results:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
