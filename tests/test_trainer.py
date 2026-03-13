from literec.data.dataset import Dataset
from literec.model.bpr import BPR
from literec.training.trainer import Trainer


def test_trainer_fit_returns_results(tiny_csv):
    ds = Dataset(tiny_csv)
    model = BPR(ds, emb_size=8)
    trainer = Trainer(model, ds, epochs=2, device="cpu", topk=[5])
    result = trainer.fit()
    assert "Recall@5" in result
    assert "NDCG@5" in result
    assert isinstance(result["Recall@5"], float)


def test_trainer_device_auto():
    from literec.training.trainer import _resolve_device
    device = _resolve_device("auto")
    assert device in ("cpu", "cuda", "mps")


def test_trainer_seed_reproducibility(tiny_csv):
    ds = Dataset(tiny_csv)
    model1 = BPR(ds, emb_size=8)
    trainer1 = Trainer(model1, ds, epochs=2, device="cpu", seed=42, topk=[5])
    result1 = trainer1.fit()

    model2 = BPR(ds, emb_size=8)
    trainer2 = Trainer(model2, ds, epochs=2, device="cpu", seed=42, topk=[5])
    result2 = trainer2.fit()

    assert result1["Recall@5"] == result2["Recall@5"]
