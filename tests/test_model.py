"""PE-RankFormer model: forward pass, output range, checkpointing (task spec §40)."""

from __future__ import annotations

import pandas as pd
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
