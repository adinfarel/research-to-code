import numpy as np
import pytest

from src.transformers.attention.attention import (
    MultiHeadAttention,
    GroupQueryAttention,
    MultiLatentAttention,
    MultiQueryAttention,
    _matmul_2d,
    _matmul_4d,
    _softmax,
)

class TestUTILS:
    def test_matmul_2d_known_value(self):
        mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        # [1*5+2*7, 1*6+2*8] = [19, 22]
        # [3*5+4*7, 3*6+4*8] = [43, 50]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])

        result = _matmul_2d(mat1, mat2)
        np.testing.assert_allclose(result, expected)


    def test_matmul_2d_shape_mismatch_raises(self):
        mat1 = np.random.randn(2, 3)
        mat2 = np.random.randn(4, 2)
        try:
            _matmul_2d(mat1, mat2)
            assert False, "Expected ValueError"
        except ValueError:
            pass


    def test_matmul_4d_matches_numpy_matmul(self):
        np.random.seed(0)
        t1 = np.random.randn(2, 3, 4, 5)
        t2 = np.random.randn(2, 3, 5, 6)

        result = _matmul_4d(t1, t2)
        expected = np.matmul(t1, t2)  

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


    def test_softmax_sums_to_one(self):
        x = np.random.randn(2, 4)
        result = _softmax(x, axis=-1)

        np.testing.assert_allclose(result.sum(axis=-1), np.ones(2), rtol=1e-6)


    def test_softmax_numerically_stable_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = _softmax(x, axis=-1)

        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(axis=-1), [1.0], rtol=1e-6)

class TestMHA:
    def test_mha_output_shape(self):
        B, T, C, n_head = 2, 5, 16, 4
        X = np.random.randn(B, T, C)

        mha = MultiHeadAttention(embed_dim=C, n_head=n_head)
        out = mha(X)

        assert out.shape == (B, T, C)


    def test_mha_causal_mask_blocks_future_tokens(self):
        np.random.seed(1)
        B, T, C, n_head = 1, 4, 8, 2

        mha = MultiHeadAttention(embed_dim=C, n_head=n_head)

        X1 = np.random.randn(B, T, C)
        X2 = X1.copy()
        X2[:, -1, :] = np.random.randn(C) * 100 

        out1 = mha(X1, causal_mask=True)
        out2 = mha(X2, causal_mask=True)

        np.testing.assert_allclose(out1[:, 0, :], out2[:, 0, :], rtol=1e-6, atol=1e-6)


    def test_mha_no_causal_mask_all_tokens_see_each_other(self):
        np.random.seed(2)
        B, T, C, n_head = 1, 4, 8, 2

        mha = MultiHeadAttention(embed_dim=C, n_head=n_head)

        X1 = np.random.randn(B, T, C)
        X2 = X1.copy()
        X2[:, -1, :] = np.random.randn(C) * 100

        out1 = mha(X1, causal_mask=False)
        out2 = mha(X2, causal_mask=False)

        assert not np.allclose(out1[:, 0, :], out2[:, 0, :])


    def test_mha_kv_cache_matches_full_forward(self):
        np.random.seed(3)
        B, C, n_head = 1, 8, 2

        mha_full = MultiHeadAttention(embed_dim=C, n_head=n_head)
        mha_cached = MultiHeadAttention(embed_dim=C, n_head=n_head)

        mha_cached.query = mha_full.query.copy()
        mha_cached.key = mha_full.key.copy()
        mha_cached.value = mha_full.value.copy()
        mha_cached.proj = mha_full.proj.copy()

        T_total = 4
        X_full = np.random.randn(B, T_total, C)

        out_full = mha_full(X_full, causal_mask=True, use_cache=False)

        mha_cached.reset_cache()
        outputs_step = []
        for t in range(T_total):
            x_step = X_full[:, t:t + 1, :] 
            out_step = mha_cached(x_step, causal_mask=True, use_cache=True)
            outputs_step.append(out_step)

        out_cached = np.concatenate(outputs_step, axis=1)

        np.testing.assert_allclose(out_full, out_cached, rtol=1e-5, atol=1e-5)


    def test_mha_reset_cache_clears_state(self):
        C, n_head = 8, 2
        mha = MultiHeadAttention(embed_dim=C, n_head=n_head)

        X = np.random.randn(1, 2, C)
        mha(X, use_cache=True)

        assert mha.k_cache is not None

        mha.reset_cache()
        assert mha.k_cache is None
        assert mha.v_cache is None


class TestMQA:

    def test_output_shape(self):
        B, T, C, n_head = 2, 5, 16, 4
        X = np.random.randn(B, T, C)

        mqa = MultiQueryAttention(embed_dim=C, n_head=n_head)
        out = mqa(X)

        assert out.shape == (B, T, C)

    def test_kv_projection_produces_single_head(self):
        C, n_head = 16, 4
        head_dim = C // n_head

        mqa = MultiQueryAttention(embed_dim=C, n_head=n_head)

        assert mqa.key.shape == (C, head_dim)
        assert mqa.value.shape == (C, head_dim)

    def test_kv_cache_matches_full_forward(self):
        np.random.seed(4)
        B, C, n_head = 1, 8, 2

        mqa_full = MultiQueryAttention(embed_dim=C, n_head=n_head)
        mqa_cached = MultiQueryAttention(embed_dim=C, n_head=n_head)

        mqa_cached.query = mqa_full.query.copy()
        mqa_cached.key = mqa_full.key.copy()
        mqa_cached.value = mqa_full.value.copy()
        mqa_cached.proj = mqa_full.proj.copy()

        T_total = 4
        X_full = np.random.randn(B, T_total, C)

        out_full = mqa_full(X_full, causal_mask=True, use_cache=False)

        mqa_cached.reset_cache()
        outputs_step = []
        for t in range(T_total):
            out_step = mqa_cached(X_full[:, t:t + 1, :], causal_mask=True, use_cache=True)
            outputs_step.append(out_step)

        out_cached = np.concatenate(outputs_step, axis=1)
        np.testing.assert_allclose(out_full, out_cached, rtol=1e-5, atol=1e-5)


class TestGQA:

    def test_output_shape(self):
        B, T, C, n_head, n_kv_head = 2, 5, 16, 4, 2
        X = np.random.randn(B, T, C)

        gqa = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=n_kv_head)
        out = gqa(X)

        assert out.shape == (B, T, C)

    def test_kv_projection_sized_to_n_kv_head(self):
        C, n_head, n_kv_head = 16, 4, 2
        head_dim = C // n_head

        gqa = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=n_kv_head)

        assert gqa.key.shape == (C, n_kv_head * head_dim)
        assert gqa.value.shape == (C, n_kv_head * head_dim)

    def test_equals_mha_when_n_kv_head_equals_n_head(self):
        C, n_head = 16, 4
        gqa = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=n_head)

        assert gqa.num_queries_each_kv_head == 1

        X = np.random.randn(1, 3, C)
        out = gqa(X)
        assert out.shape == (1, 3, C)

    def test_equals_mqa_when_n_kv_head_is_one(self):
        C, n_head = 16, 4
        gqa = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=1)

        assert gqa.num_queries_each_kv_head == n_head

        X = np.random.randn(1, 3, C)
        out = gqa(X)
        assert out.shape == (1, 3, C)

    def test_kv_cache_matches_full_forward(self):
        np.random.seed(5)
        B, C, n_head, n_kv_head = 1, 16, 4, 2

        gqa_full = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=n_kv_head)
        gqa_cached = GroupQueryAttention(embed_dim=C, n_head=n_head, n_kv_head=n_kv_head)

        gqa_cached.query = gqa_full.query.copy()
        gqa_cached.key = gqa_full.key.copy()
        gqa_cached.value = gqa_full.value.copy()
        gqa_cached.proj = gqa_full.proj.copy()

        T_total = 4
        X_full = np.random.randn(B, T_total, C)

        out_full = gqa_full(X_full, causal_mask=True, use_cache=False)

        gqa_cached.reset_cache()
        outputs_step = []
        for t in range(T_total):
            out_step = gqa_cached(X_full[:, t:t + 1, :], causal_mask=True, use_cache=True)
            outputs_step.append(out_step)

        out_cached = np.concatenate(outputs_step, axis=1)
        np.testing.assert_allclose(out_full, out_cached, rtol=1e-5, atol=1e-5)


class TestMLA:

    def test_output_shape(self):
        B, T, C, n_head = 2, 5, 16, 4
        X = np.random.randn(B, T, C)

        mla = MultiLatentAttention(embed_dim=C, n_head=n_head,
                                    q_latent_size=12, kv_latent_size=6)
        out = mla(X)

        assert out.shape == (B, T, C)

    def test_constructor_does_not_crash(self):
        mla = MultiLatentAttention(embed_dim=16, n_head=4,
                                    q_latent_size=12, kv_latent_size=6)
        assert mla.q_latent_size == 12
        assert mla.kv_latent_size == 6

    def test_kv_cache_stores_compressed_latent_not_full_kv(self):
        C, n_head, kv_latent_size = 16, 4, 6

        mla = MultiLatentAttention(embed_dim=C, n_head=n_head,
                                    q_latent_size=12, kv_latent_size=kv_latent_size)

        X = np.random.randn(1, 3, C)
        mla(X, use_cache=True)

        assert mla.k_cache.shape[-1] == kv_latent_size  # type: ignore
        assert mla.k_cache.shape[-1] < C  # type: ignore 

    def test_kv_cache_matches_full_forward(self):
        np.random.seed(6)
        B, C, n_head = 1, 16, 4

        mla_full = MultiLatentAttention(embed_dim=C, n_head=n_head,
                                         q_latent_size=12, kv_latent_size=6)
        mla_cached = MultiLatentAttention(embed_dim=C, n_head=n_head,
                                           q_latent_size=12, kv_latent_size=6)

        mla_cached.q_down_proj = mla_full.q_down_proj.copy()
        mla_cached.q_up_proj = mla_full.q_up_proj.copy()
        mla_cached.kv_down_proj = mla_full.kv_down_proj.copy()
        mla_cached.k_up_proj = mla_full.k_up_proj.copy()
        mla_cached.v_up_proj = mla_full.v_up_proj.copy()
        mla_cached.proj = mla_full.proj.copy()

        T_total = 4
        X_full = np.random.randn(B, T_total, C)

        out_full = mla_full(X_full, causal_mask=True, use_cache=False)

        mla_cached.reset_cache()
        outputs_step = []
        for t in range(T_total):
            out_step = mla_cached(X_full[:, t:t + 1, :], causal_mask=True, use_cache=True)
            outputs_step.append(out_step)

        out_cached = np.concatenate(outputs_step, axis=1)
        np.testing.assert_allclose(out_full, out_cached, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    import sys
    
    sys.exit(pytest.main([__file__, "-v"]))