use crate::self_attention::SelfAttention;
use crate::feed_forward::FeedForward;
use crate::llm::{LayerNorm, Layer};
use ndarray::Array2;
pub struct TransformerBlock {
    attention: SelfAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            layer_norm1: LayerNorm::new(embedding_dim),
            layer_norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let attention_out = self.attention.forward_with_residual(input, &self.layer_norm1);
        self.feed_forward.forward_with_residual(&attention_out, &self.layer_norm2)
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_after_ffn = self.feed_forward.backward(grads, lr);
        let grad_after_attn = self.attention.backward(&grad_after_ffn, lr);

        grad_after_attn
    }   
}
