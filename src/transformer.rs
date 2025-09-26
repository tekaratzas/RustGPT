use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;
use crate::llm::Layer;
use crate::self_attention::SelfAttention;
use ndarray::Array2;
pub struct TransformerBlock {
    attention: SelfAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm, // After attention
    norm2: LayerNorm, // After feed forward
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Standard Transformer architecture: attention + norm -> feedforward + norm
        let attention_out = self.attention.forward(input); // includes residual
        let norm1_out = self.norm1.normalize(&attention_out);

        let feed_forward_out = self.feed_forward.forward(&norm1_out); // includes residual
        let norm2_out = self.norm2.normalize(&feed_forward_out);

        norm2_out
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backward through second LayerNorm
        let grad_norm2 = self.norm2.backward(grads, lr);

        // Backward through feed-forward (includes residual connection)
        let grad_ffn = self.feed_forward.backward(&grad_norm2, lr);

        // Backward through first LayerNorm
        let grad_norm1 = self.norm1.backward(&grad_ffn, lr);

        // Backward through attention (includes residual connection)
        let grad_attn = self.attention.backward(&grad_norm1, lr);

        grad_attn
    }

    fn parameters(&self) -> usize {
        self.attention.parameters() + self.feed_forward.parameters() + self.norm1.parameters() + self.norm2.parameters()
    }
}
