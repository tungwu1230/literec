from literec.config import LiteRecConfig


def test_default_config():
    cfg = LiteRecConfig()
    assert cfg.split == "loo"
    assert cfg.min_interactions == 5
    assert cfg.device == "auto"
    assert cfg.topk == [10, 20]


def test_config_override():
    cfg = LiteRecConfig(split="random", lr=0.01, topk=[5, 10, 20])
    assert cfg.split == "random"
    assert cfg.lr == 0.01
    assert cfg.topk == [5, 10, 20]
