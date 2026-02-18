package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.List;

/**
 * Common interface for attention mechanisms.
 * <p>
 * Implemented by:
 * - CausalSelfAttention (single head)
 * - MultiHeadCausalSelfAttention (multiple heads)
 */
public interface Attention {
    Value[][] forward(Value[][] input);

    List<Value> parameters();
}
