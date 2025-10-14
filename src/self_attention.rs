use crate::adam::Adam;
use crate::EMBEDDING_DIM;
use ndarray::Array2;
use rand_distr::{Normal, Distribution};
use crate::llm::Layer;
use std::f32;

pub struct SelfAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    w_q: Array2<f32>, // Weight matrices for Q, K, V
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>, // Output projection matrix

    cached_input: Option<Array2<f32>>,
    cached_q: Option<Array2<f32>>,
    cached_k: Option<Array2<f32>>,
    cached_v: Option<Array2<f32>>,
    cached_attn_weights: Option<Vec<Array2<f32>>>,

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
    optimizer_w_o: Adam,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM, 8) // 8 attention heads by default
    }
}
    

impl SelfAttention {
     /// Initializes a Multi-Head Attention with random Q, K, V, O weights
     /// num_heads: Number of attention heads (embedding_dim must be divisible by num_heads)
     pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        assert!(
            embedding_dim % num_heads == 0,
            "embedding_dim must be divisible by num_heads"
        );
        
        let head_dim = embedding_dim / num_heads;
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        SelfAttention {
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

    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q); // Q = X * W_Q
        let k = input.dot(&self.w_k); // K = X * W_K
        let v = input.dot(&self.w_v); // V = X * W_V
        (q, k, v)

    }
    /// Split the input tensor into multiple heads
    /// Input shape: [seq_len, embedding_dim]
    /// Output shape: [num_heads, seq_len, head_dim]
    fn split_heads(&self, x: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut heads = Vec::with_capacity(self.num_heads);
            
        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            let head = x.slice(ndarray::s![.., start..end]).to_owned();
            heads.push(head);
        }
            
        heads
    }

    /// Concatenate multiple heads back together
    /// Input: Vec of [seq_len, head_dim] arrays
    /// Output shape: [seq_len, embedding_dim]
    fn concat_heads(&self, heads: Vec<Array2<f32>>) -> Array2<f32> {
        let seq_len = heads[0].shape()[0];
        let mut result = Array2::zeros((seq_len, self.embedding_dim));

        for (h, head) in heads.iter().enumerate() {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            result.slice_mut(ndarray::s![.., start..end]).assign(head);
        }

        result
    }

    /// Single-head attention computation
    fn attention_head(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let dk = (self.head_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Apply causal masking - prevent attention to future tokens
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let weights = self.softmax(&scores);
        let output = weights.dot(v);
        (output, weights)
    }

    /// Multi-head attention: applies attention independently for each head
    fn multi_head_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Split into heads
        let q_heads = self.split_heads(q);
        let k_heads = self.split_heads(k);
        let v_heads = self.split_heads(v);

        // Apply attention for each head
        let mut output_heads = Vec::with_capacity(self.num_heads);
        let mut attn_weights = Vec::with_capacity(self.num_heads);
        
        for i in 0..self.num_heads {
            let (head_output, head_weights) = self.attention_head(&q_heads[i], &k_heads[i], &v_heads[i]);
            output_heads.push(head_output);
            attn_weights.push(head_weights);
        }

        // Concatenate heads
        let concat_output = self.concat_heads(output_heads);
        
        (concat_output, attn_weights)
    }

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

    fn softmax_backward(
        softmax_output: &Array2<f32>,  // shape: [seq_len, vocab_size]
        grad_output: &Array2<f32>,     // shape: [seq_len, vocab_size]
    ) -> Array2<f32> {
        let mut grad_input = softmax_output.clone(); // to hold the result
    
        for ((mut grad_row, softmax_row), grad_out_row) in
            grad_input
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
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "MUltiHeadSelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Compute Q, K, V projections
        let (q, k, v) = self.compute_qkv(input);
        
        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_q = Some(q.clone());
        self.cached_k = Some(k.clone());
        self.cached_v = Some(v.clone());
        
        // Apply multi-head attention
        let (multi_head_output, attn_weights) = self.multi_head_attention(&q, &k, &v);
        self.cached_attn_weights = Some(attn_weights);
        
        // Apply output projection
        let projected = multi_head_output.dot(&self.w_o);
        
        // Add residual connection
        projected + input
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let q = self.cached_q.as_ref().unwrap();
        let k = self.cached_k.as_ref().unwrap();
        let v = self.cached_v.as_ref().unwrap();
        let attn_weights = self.cached_attn_weights.as_ref().unwrap();

        // Gradient through output projection: ∂L/∂W_o
        let multi_head_output = {
            let v_heads = self.split_heads(v);
            let mut output_heads = Vec::with_capacity(self.num_heads);
            for i in 0..self.num_heads {
                let output = attn_weights[i].dot(&v_heads[i]);
                output_heads.push(output);
            }
            self.concat_heads(output_heads)
        };

        let grad_w_o = multi_head_output.t().dot(grads);
        let grad_multi_head_output = grads.dot(&self.w_o.t());

        // Split gradient back into heads
        let grad_output_heads = self.split_heads(&grad_multi_head_output);

        // Backward through each attention head
        let q_heads = self.split_heads(q);
        let k_heads = self.split_heads(k);
        let v_heads = self.split_heads(v);

        let mut grad_q_heads = Vec::with_capacity(self.num_heads);
        let mut grad_k_heads = Vec::with_capacity(self.num_heads);
        let mut grad_v_heads = Vec::with_capacity(self.num_heads);

        for i in 0..self.num_heads {
            let grad_head = &grad_output_heads[i];
            let weights = &attn_weights[i];
            let q_head = &q_heads[i];
            let k_head = &k_heads[i];
            let v_head = &v_heads[i];

            // Gradient w.r.t. V
            let grad_v_head = weights.t().dot(grad_head);

            // Gradient w.r.t. attention weights
            let grad_attn_weights = grad_head.dot(&v_head.t());

            // Gradient through softmax
            let grad_scores = SelfAttention::softmax_backward(weights, &grad_attn_weights);

            // Scale factor for attention
            let scale = (self.head_dim as f32).sqrt();

            // Gradient w.r.t. Q and K
            let grad_q_head = grad_scores.dot(k_head) / scale;
            let grad_k_head = grad_scores.t().dot(q_head) / scale;

            grad_q_heads.push(grad_q_head);
            grad_k_heads.push(grad_k_head);
            grad_v_heads.push(grad_v_head);
        }

        // Concatenate head gradients
        let grad_q = self.concat_heads(grad_q_heads);
        let grad_k = self.concat_heads(grad_k_heads);
        let grad_v = self.concat_heads(grad_v_heads);
        
        // Gradients w.r.t. weight matrices
        let grad_w_q = input.t().dot(&grad_q);
        let grad_w_k = input.t().dot(&grad_k);
        let grad_w_v = input.t().dot(&grad_v);
        
        // Gradients w.r.t. weight matrices
        let grad_input_attention = grad_q.dot(&self.w_q.t())
            + grad_k.dot(&self.w_k.t())
            + grad_v.dot(&self.w_v.t());

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
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }
}