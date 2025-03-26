use crate::self_attention::SelfAttention;
use crate::feed_forward::FeedForward;
use crate::llm::Layer;
use ndarray::Array2;
pub struct TransformerBlock {
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let attention_out = self.attention.forward(input);
        let feed_forward_out = self.feed_forward.forward(&attention_out);
        feed_forward_out
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_after_ffn = self.feed_forward.backward(grads, lr);
        let grad_after_attn = self.attention.backward(&grad_after_ffn, lr);

        grad_after_attn
    }   
}
