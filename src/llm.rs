use crate::EMBEDDING_DIM;
use crate::Embeddings;
use crate::HIDDEN_DIM;
use crate::MAX_SEQ_LEN;
use crate::Vocab;
use crate::output_projection::OutputProjection;
use crate::transformer::TransformerBlock;
use ndarray::{Array1, Array2, Axis};
use std::cmp::Ordering;

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;
}

pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
}

impl Default for LLM {
    fn default() -> Self {
        let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        Self {
            vocab: Vocab::default(),
            network: vec![
                Box::new(Embeddings::default()),
                Box::new(transformer_block),
                Box::new(output_projection),
            ],
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self { vocab, network }
    }
}

impl LLM {
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|layer| layer.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        // Sum the parameters across all layers in the network
        self.network
            .iter()
            .map(|layer: &Box<dyn Layer>| layer.parameters())
            .sum::<usize>()
    }

    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);

        // Handle empty output
        if output_tokens.is_empty() {
            return String::new();
        }

        // Convert token_ids to strings
        let token_strs = output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>();

        token_strs.join(" ")
    }

    fn forward(&mut self, text: &str) -> Vec<usize> {
        // Tokenize the input text
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        // Safety check: ensure we have at least one token
        if tokenized.is_empty() {
            return output_tokens;
        }

        let input_len = tokenized.len();

        // Prevent overflow if input_len >= MAX_SEQ_LEN
        if input_len >= MAX_SEQ_LEN {
            return output_tokens;
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            // let tokenized_clone = tokenized.clone();

            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }

            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            // Safety check: ensure we have at least one token
            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            // Softmax - convert activiations of each token to a probability distribution over the vocabulary
            let probs = Self::softmax(&last_logit); // 1 x vocab_size

            // Greedy Decode - Choose the highest probability token for each position
            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        output_tokens
    }

    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) {
        let tokenized_data = data
            .iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // 1. Slice input and targets
                let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
                let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab. 

                // Forward pass
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let probs = Self::softmax(&logits);

                total_loss += Self::cross_entropy_loss_step(&probs, target_ids);

                // Backward pass
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids); // this is d_L/d_output_projection

                // Apply gradient clipping BEFORE backpropagation
                Self::clip_gradients(&mut grads_output, 5.0);

                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, lr);
                }

                let tokens = Self::greedy_decode(&probs);
                let next_token = tokens[tokens.len() - 1];

                if next_token == self.vocab.encode("</s>").unwrap() {
                    continue;
                }
            }

            println!(
                "Epoch {}: Loss = {:.4}",
                epoch,
                total_loss / tokenized_data.len() as f32
            );
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Split by whitespace first
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            // Special case for end token
            if word == "</s>" {
                if let Some(token_id) = self.vocab.encode(word) {
                    tokens.push(token_id);
                }
                continue;
            }

            let mut current_word = String::new();

            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    // If we have a word before the punctuation, add it
                    if !current_word.is_empty() {
                        if let Some(token_id) = self.vocab.encode(&current_word) {
                            tokens.push(token_id);
                        }
                        current_word.clear();
                    }

                    // Add the punctuation as its own token
                    if let Some(token_id) = self.vocab.encode(&c.to_string()) {
                        tokens.push(token_id);
                    }
                } else {
                    current_word.push(c);
                }
            }

            // Add any remaining word
            if !current_word.is_empty() {
                if let Some(token_id) = self.vocab.encode(&current_word) {
                    tokens.push(token_id);
                }
            }
        }

        tokens
    }

    fn softmax(logits: &Array2<f32>) -> Array2<f32> {
        // logits is seq_len x vocab_size
        let mut result = logits.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            // Calculate exp for each element
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .to_vec()
    }

    fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        let mut loss = 0.0;
        for row_idx in 0..probs.shape()[0] {
            let prob_target = probs[[row_idx, target[row_idx]]]; // Get probability of correct token
            loss -= prob_target.max(1e-15).ln(); // Add numerical stability
        }

        loss / target.len() as f32
    }

    fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone(); // Start with softmax probabilities

        if probs.shape()[0] != target.len() {
            panic!("Probs and target must have the same number of rows");
        }

        let batch_size = target.len() as f32;

        // Compute correct softmax + cross-entropy gradient: softmax - one_hot(target)
        for row_idx in 0..grads.shape()[0] {
            grads[[row_idx, target[row_idx]]] -= 1.0; // Convert to: p - y (where y is one-hot)
        }

        // Normalize by batch size for stable training
        grads.mapv_inplace(|x| x / batch_size);

        grads
    }

    fn clip_gradients(grads: &mut Array2<f32>, max_norm: f32) {
        // Calculate L2 norm of gradients
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // If norm exceeds max_norm, scale gradients down
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }
}
