from literec import Dataset, BPR, LightGCN, NGCF, Trainer


def test_e2e_bpr(tiny_csv):
    ds = Dataset(tiny_csv)
    model = BPR(ds, emb_size=8)
    trainer = Trainer(model, ds, epochs=3, device="cpu", topk=[2])
    result = trainer.fit()
    assert "Recall@2" in result


def test_e2e_lightgcn(tiny_csv):
    ds = Dataset(tiny_csv)
    model = LightGCN(ds, emb_size=8, n_layers=2)
    trainer = Trainer(model, ds, epochs=3, device="cpu", topk=[2])
    result = trainer.fit()
    assert "Recall@2" in result


def test_e2e_ngcf(tiny_csv):
    ds = Dataset(tiny_csv)
    model = NGCF(ds, emb_size=8, n_layers=2, dropout=0.1)
    trainer = Trainer(model, ds, epochs=3, device="cpu", topk=[2])
    result = trainer.fit()
    assert "Recall@2" in result
