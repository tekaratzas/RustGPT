use std::f32;

use ndarray::Array2;
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, adam::Adam, llm::Layer};

/// Multi-Head Self-Attention implementation
///
/// This layer splits the embedding dimension into multiple attention heads,
/// allowing the model to attend to information from different representation
/// subspaces at different positions.
///
/// Architecture:
/// - Input: [seq_len, embedding_dim]
/// - Split into num_heads with head_dim = embedding_dim / num_heads
/// - Each head computes its own Q, K, V and attention
/// - Outputs are concatenated and projected through W_o
pub struct MultiHeadAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    // Weight matrices for Q, K, V projections
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    // Output projection matrix
    w_o: Array2<f32>,

    // Cache for backward pass
    cached_input: Option<Array2<f32>>,
    cached_q: Option<Array2<f32>>,
    cached_k: Option<Array2<f32>>,
    cached_v: Option<Array2<f32>>,
    cached_attn_weights: Option<Vec<Array2<f32>>>,

    // Optimizers for each weight matrix
    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
    optimizer_w_o: Adam,
}

impl Default for MultiHeadAttention {
    fn default() -> Self {
        MultiHeadAttention::new(EMBEDDING_DIM, 8)
    }
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention layer
    ///
    /// # Arguments
    /// * `embedding_dim` - The dimension of input embeddings
    /// * `num_heads` - Number of attention heads (must divide embedding_dim evenly)
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        assert_eq!(
            embedding_dim % num_heads,
            0,
            "embedding_dim must be divisible by num_heads"
        );

        let head_dim = embedding_dim / num_heads;
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        MultiHeadAttention {
            embedding_dim,
            num_heads,
            head_dim,
            w_q: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_o: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn_weights: None,
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_o: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    /// Computes Q, K, V projections from input
    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q); // Q = X * W_Q
        let k = input.dot(&self.w_k); // K = X * W_K
        let v = input.dot(&self.w_v); // V = X * W_V
        (q, k, v)
    }

    /// Splits the input into multiple heads
    ///
    /// # Arguments
    /// * `x` - Input of shape [seq_len, embedding_dim]
    ///
    /// # Returns
    /// Vector of arrays, one per head, each of shape [seq_len, head_dim]
    pub fn split_heads(&self, x: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut heads = Vec::new();

        for h in 0..self.num_heads {
            let start_idx = h * self.head_dim;
            let end_idx = start_idx + self.head_dim;
            let head = x.slice(ndarray::s![.., start_idx..end_idx]).to_owned();
            heads.push(head);
        }

        heads
    }

    /// Concatenates multiple heads back into a single array
    ///
    /// # Arguments
    /// * `heads` - Vector of arrays, one per head, each of shape [seq_len, head_dim]
    ///
    /// # Returns
    /// Array of shape [seq_len, embedding_dim]
    pub fn concat_heads(&self, heads: &[Array2<f32>]) -> Array2<f32> {
        let seq_len = heads[0].shape()[0];
        let mut result = Array2::zeros((seq_len, self.embedding_dim));

        for (h, head) in heads.iter().enumerate() {
            let start_idx = h * self.head_dim;
            let end_idx = start_idx + self.head_dim;
            result
                .slice_mut(ndarray::s![.., start_idx..end_idx])
                .assign(head);
        }

        result
    }

    /// Applies softmax function row-wise
    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    /// Computes gradient of softmax function
    fn softmax_backward(
        softmax_output: &Array2<f32>, // shape: [seq_len, seq_len]
        grad_output: &Array2<f32>,    // shape: [seq_len, seq_len]
    ) -> Array2<f32> {
        let mut grad_input = softmax_output.clone();

        for ((mut grad_row, softmax_row), grad_out_row) in grad_input
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            // dot product: y ⊙ dL/dy
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();

            for ((g, &y_i), &dy_i) in grad_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }

        grad_input
    }

    /// Performs attention for a single head
    fn attention_head(
        &self,
        q_head: &Array2<f32>,
        k_head: &Array2<f32>,
        v_head: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let dk = (self.head_dim as f32).sqrt();
        let k_t = k_head.t();
        let mut scores = q_head.dot(&k_t) / dk;

        // Apply causal masking
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let attn_weights = self.softmax(&scores);
        let output = attn_weights.dot(v_head);

        (output, attn_weights)
    }
}

impl Layer for MultiHeadAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        // Compute Q, K, V projections
        let (q, k, v) = self.compute_qkv(input);
        self.cached_q = Some(q.clone());
        self.cached_k = Some(k.clone());
        self.cached_v = Some(v.clone());

        // Split into heads
        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        // Apply attention for each head
        let mut head_outputs = Vec::new();
        let mut attn_weights = Vec::new();
        for i in 0..self.num_heads {
            let (head_output, head_weights) =
                self.attention_head(&q_heads[i], &k_heads[i], &v_heads[i]);
            head_outputs.push(head_output);
            attn_weights.push(head_weights);
        }
        self.cached_attn_weights = Some(attn_weights);

        // Concatenate heads
        let concat = self.concat_heads(&head_outputs);

        // Apply output projection
        let output = concat.dot(&self.w_o);

        // Add residual connection
        output + input
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let q = self.cached_q.as_ref().unwrap();
        let k = self.cached_k.as_ref().unwrap();
        let v = self.cached_v.as_ref().unwrap();
        let attn_weights = self.cached_attn_weights.as_ref().unwrap();

        // Gradient through residual connection
        let grad_output = grads.clone();

        // Gradient through output projection: dL/dW_o = concat^T @ grad_output
        let q_heads = self.split_heads(q);
        let k_heads = self.split_heads(k);
        let v_heads = self.split_heads(v);

        // Recompute head outputs for gradient calculation
        let mut head_outputs = Vec::new();
        for i in 0..self.num_heads {
            let head_output = attn_weights[i].dot(&v_heads[i]);
            head_outputs.push(head_output);
        }
        let concat = self.concat_heads(&head_outputs);

        // Gradient of W_o
        let grad_w_o = concat.t().dot(&grad_output);

        // Gradient w.r.t. concatenated heads
        let grad_concat = grad_output.dot(&self.w_o.t());

        // Split gradient back into heads
        let grad_heads = self.split_heads(&grad_concat);

        // Backpropagate through each attention head
        let mut grad_q_heads = Vec::new();
        let mut grad_k_heads = Vec::new();
        let mut grad_v_heads = Vec::new();

        for i in 0..self.num_heads {
            let dk = (self.head_dim as f32).sqrt();

            // Gradient w.r.t. V: dL/dV = attn_weights^T @ grad_head
            let grad_v_head = attn_weights[i].t().dot(&grad_heads[i]);

            // Gradient w.r.t. attention weights
            let grad_attn_weights = grad_heads[i].dot(&v_heads[i].t());

            // Gradient w.r.t. scores (through softmax)
            let grad_scores = Self::softmax_backward(&attn_weights[i], &grad_attn_weights);

            // Gradient w.r.t. Q and K
            let grad_q_head = grad_scores.dot(&k_heads[i]) / dk;
            let grad_k_head = grad_scores.t().dot(&q_heads[i]) / dk;

            grad_q_heads.push(grad_q_head);
            grad_k_heads.push(grad_k_head);
            grad_v_heads.push(grad_v_head);
        }

        // Concatenate head gradients
        let grad_q = self.concat_heads(&grad_q_heads);
        let grad_k = self.concat_heads(&grad_k_heads);
        let grad_v = self.concat_heads(&grad_v_heads);

        // Gradient w.r.t. weight matrices
        let grad_w_q = input.t().dot(&grad_q);
        let grad_w_k = input.t().dot(&grad_k);
        let grad_w_v = input.t().dot(&grad_v);

        // Gradient w.r.t. input (through Q, K, V projections)
        let grad_input_attention =
            grad_q.dot(&self.w_q.t()) + grad_k.dot(&self.w_k.t()) + grad_v.dot(&self.w_v.t());

        // Add gradient from residual connection
        let grad_input = grad_input_attention + grads;

        // Update weights using Adam optimizer
        self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
        self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
        self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);
        self.optimizer_w_o.step(&mut self.w_o, &grad_w_o, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_and_concat_heads() {
        let embedding_dim = 128;
        let num_heads = 8;
        let seq_len = 10;

        let mha = MultiHeadAttention::new(embedding_dim, num_heads);
        let input = Array2::ones((seq_len, embedding_dim));

        // Test split
        let heads = mha.split_heads(&input);
        assert_eq!(heads.len(), num_heads);
        for head in &heads {
            assert_eq!(head.shape(), [seq_len, mha.head_dim]);
        }

        // Test concat
        let concat = mha.concat_heads(&heads);
        assert_eq!(concat.shape(), [seq_len, embedding_dim]);

        // Verify split and concat are inverses
        for i in 0..seq_len {
            for j in 0..embedding_dim {
                assert!((concat[[i, j]] - input[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_multihead_attention_shapes() {
        let embedding_dim = 128;
        let num_heads = 8;
        let seq_len = 10;

        let mut mha = MultiHeadAttention::new(embedding_dim, num_heads);
        let input = Array2::ones((seq_len, embedding_dim));

        let output = mha.forward(&input);
        assert_eq!(output.shape(), [seq_len, embedding_dim]);
    }

    #[test]
    fn test_multihead_attention_parameter_count() {
        let embedding_dim = 128;
        let num_heads = 8;

        let mha = MultiHeadAttention::new(embedding_dim, num_heads);

        // Should have 4 weight matrices: W_q, W_k, W_v, W_o
        // Each is embedding_dim x embedding_dim
        let expected = 4 * embedding_dim * embedding_dim;
        assert_eq!(mha.parameters(), expected);
    }
}
