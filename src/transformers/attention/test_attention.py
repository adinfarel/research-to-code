import numpy as np
import pytest

from src.transformers.attention.attention import (
    MultiHeadAttention,
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

if __name__ == "__main__":
    import sys
    
    sys.exit(pytest.main([__file__, "-v"]))