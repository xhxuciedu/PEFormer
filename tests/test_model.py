"""PE-RankFormer model: forward pass, output range, checkpointing (task spec §40)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from pe_rankformer.data.context import ContextVocab
from pe_rankformer.data.dataset import PEDataset, collate, featurize
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig


def _toy_corpus(n=12):
    rows = []
    for i in range(n):
        rows.append(
            {
                "full_unedited": "ACGTACGTACGTACGTACGT",
                "full_edited": "ACGTACGTACCTACGTACGT" if i % 2 == 0 else "ACGTACGTACGTACGTACGT",
                "spacer": "ACGTACGTACGTACGTACGT",
                "pbs": "ACGTACGTACGTA",
                "rtt": "ACGTACGTACGTACGTACGT",
                "cell_type": "HEK293T" if i < 6 else "HeLa",
                "pe_type": "PE2",
                "cas9_type": "PEmax-Cas9",
                "cas9_pam": "SpNGG",
                "scaffold_name": "BlpI_F+E",
                "motif": "tevoPreQ1",
                "source_study": "hsu2026",
                "edited": (i % 7) / 7,
                "fold": i % 5,
                "record_id": f"r{i}",
            }
        )
    return pd.DataFrame(rows)


def _tiny_model() -> tuple[PERankFormer, ContextVocab]:
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    cfg = PERankFormerConfig(
        d_model=32,
        n_heads=2,
        ffn_dim=64,
        n_edit_layers=1,
        n_peg_layers=1,
        n_cross_blocks=1,
        context_fields=vocab.fields,
        context_vocab_sizes=vocab.sizes(),
        context_embed_dim=8,
    )
    return PERankFormer(cfg), vocab


def test_forward_pass_shape():
    model, vocab = _tiny_model()
    df = _toy_corpus()
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    score = model(batch)
    assert score.shape == (6,)


def test_output_range_after_sigmoid():
    model, vocab = _tiny_model()
    df = _toy_corpus()
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    eff = model.predict_efficiency(batch)
    assert (eff >= 0).all() and (eff <= 1).all()


def test_checkpoint_save_load_roundtrip(tmp_path):
    model, vocab = _tiny_model()
    df = _toy_corpus()
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(4)])

    model.eval()
    with torch.no_grad():
        before = model(batch)

    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    model2 = PERankFormer(model.config)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model2.eval()
    with torch.no_grad():
        after = model2(batch)

    assert torch.allclose(before, after)


def test_gradients_flow_to_all_major_components():
    model, vocab = _tiny_model()
    df = _toy_corpus()
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])

    score = model(batch)
    score.sum().backward()

    assert model.edit_token_embed.weight.grad is not None
    assert model.peg_nuc_embed.weight.grad is not None
    assert model.cross_blocks[0].edit_attn.out_proj.weight.grad is not None
    assert model.film.net[0].weight.grad is not None
    assert model.head[1].weight.grad is not None


def test_param_count_in_target_range_for_full_size_config():
    vocab = ContextVocab.fit(_toy_corpus())
    cfg = PERankFormerConfig(context_fields=vocab.fields, context_vocab_sizes=vocab.sizes())
    model = PERankFormer(cfg)
    n = model.num_parameters()
    assert 15_000_000 <= n <= 35_000_000, f"expected ~20-30M params, got {n}"


def test_no_context_ablation_forward_pass():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=1, n_peg_layers=1, n_cross_blocks=1,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        use_context=False,
    )
    model = PERankFormer(cfg)
    assert model.context_encoder is None
    assert model.film is None
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    score = model(batch)
    assert score.shape == (6,)


def test_batch_of_one_does_not_crash():
    model, vocab = _tiny_model()
    df = _toy_corpus()
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[0]])
    model.eval()
    with torch.no_grad():
        score = model(batch)
    assert score.shape == (1,)


def test_simplex_head_shapes_and_efficiency_range():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=1, n_peg_layers=1, n_cross_blocks=1,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        outcome_head="simplex",
    )
    model = PERankFormer(cfg)
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    out = model(batch)
    assert out.shape == (6, 3)  # 3-way outcome logits
    eff = model.efficiency_from_output(out)
    assert eff.shape == (6,)
    assert (eff >= 0).all() and (eff <= 1).all()
    assert model.ranking_score(out).shape == (6,)


def test_scalar_head_ranking_score_is_identity():
    model, vocab = _tiny_model()
    out = torch.randn(5)
    assert torch.allclose(model.ranking_score(out), out)


def test_layerwise_context_forward_and_gradients():
    """Round-2 Family A: FiLM at every block instead of once after pooling."""
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=2,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        context_strategy="layerwise", outcome_head="simplex",
    )
    model = PERankFormer(cfg)
    assert model.film is None
    assert len(model.edit_films) == 2 and len(model.peg_films) == 2
    assert len(model.cross_edit_films) == 2 and len(model.cross_peg_films) == 2

    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    out = model(batch)
    assert out.shape == (6, 3)
    out.sum().backward()
    assert model.edit_token_embed.weight.grad is not None
    assert model.edit_films[0].net[0].weight.grad is not None
    assert model.cross_edit_films[0].net[0].weight.grad is not None
    assert model.peg_films[-1].net[0].weight.grad is not None


def test_layerwise_context_requires_use_context():
    vocab = ContextVocab.fit(_toy_corpus())
    with pytest.raises(ValueError):
        PERankFormerConfig(
            context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(),
            context_strategy="layerwise", use_context=False,
        )


def test_layerwise_and_late_strategies_differ_in_param_count():
    vocab = ContextVocab.fit(_toy_corpus())
    late = PERankFormer(PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=2,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        context_strategy="late",
    ))
    layerwise = PERankFormer(PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=2,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        context_strategy="layerwise",
    ))
    assert layerwise.num_parameters() > late.num_parameters()


def test_feature_branch_forward_and_gradients():
    """Round-2 Family C: continuous-feature MLP branch concatenated onto pooled repr."""
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    n_feat = 5
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=1, n_peg_layers=1, n_cross_blocks=1,
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), context_embed_dim=8,
        n_features=n_feat, feature_hidden_dim=16, outcome_head="simplex",
    )
    model = PERankFormer(cfg)
    assert model.feature_branch is not None

    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(6)])
    batch["features"] = torch.randn(6, n_feat)
    batch["features_missing"] = torch.zeros(6, n_feat)

    out = model(batch)
    assert out.shape == (6, 3)
    out.sum().backward()
    assert model.feature_branch.net[0].weight.grad is not None


def test_feature_branch_disabled_by_default():
    vocab = ContextVocab.fit(_toy_corpus())
    cfg = PERankFormerConfig(context_fields=vocab.fields, context_vocab_sizes=vocab.sizes())
    model = PERankFormer(cfg)
    assert model.feature_branch is None
