use ndarray::Array1;
use ndarray::{Array2, Axis};
use crate::transformer::TransformerBlock;
use crate::Embeddings;
use crate::Vocab;
use crate::output_projection::OutputProjection;
use crate::EMBEDDING_DIM;
use crate::HIDDEN_DIM;
use crate::MAX_SEQ_LEN;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn forward_with_residual(&mut self, input: &Array2<f32>, layer_norm: &LayerNorm) -> Array2<f32> {
        let output = self.forward(input);
        let residual = &output + input;
        layer_norm.normalize(&residual) 
    }

    // I want to use this, but some layers don't have a fixed input or output shape. It's dependent on the Sequence Length.
    // fn input_shape(&self) -> &[usize];
    // fn output_shape(&self) -> &[usize];
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
        Self {
            vocab,
            network
        }
    }
}

impl LLM {
    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);

        // Convert token_ids to strings
        let token_strs = output_tokens.iter().map(|t| self.vocab.decode[t].clone()).collect::<Vec<String>>();

        token_strs.join(" ")
    }

    fn forward(&mut self, text: &str) -> Vec<usize> {
        // Tokenize the input text
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        let input_len = tokenized.len();
                
        for _ in 0..MAX_SEQ_LEN-input_len {
            let tokenized_clone = tokenized.clone();

            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }

            let mut input: Array2<f32> = Array2::zeros((1, tokenized_clone.len()));
            input.row_mut(0).assign(&tokenized_clone.into_iter().map(|x| x as f32).collect::<Array1<f32>>());
            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;
            let last_logit = logits.row(logits.shape()[0] - 1).to_owned().insert_axis(Axis(0));

            // Softmax - convert activiations of each token to a probability distribution over the vocabulary
            let probs = Self::softmax(&last_logit); // 1 x vocab_size

            // Greedy Decode - Choose the highest probability token for each position
            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() { break; }
        }

        output_tokens
    }

    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) {
        let tokenized_data = data
            .iter()
            .map(|input| (self.tokenize(input)))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (training_row) in &tokenized_data {
                if training_row.len() < 2 { continue; }

                // 1. Slice input and targets
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                // Forward pass
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input.row_mut(0).assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let probs = Self::softmax(&logits);

                total_loss += Self::cross_entropy_loss_step(&probs, target_ids);

                // Backward pass
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);
                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, lr);
                }

                let tokens = Self::greedy_decode(&probs);
                let next_token = tokens[tokens.len() - 1];

                if next_token == self.vocab.encode("</s>").unwrap() { break; }
            }
            
            println!("Epoch {}: Loss = {:.4}", epoch, total_loss / tokenized_data.len() as f32);
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

    fn softmax(logits: &Array2<f32>) -> Array2<f32> { // logits is 1 x vocab_size
        let mut result = logits.clone();
        
        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| x.exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();
            
            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }
        
        result
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        }).to_vec()
    } 

    fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        let mut loss = 0.0;
        for row_idx in 0..probs.shape()[0] {
            let prob_target = probs[[0, target[row_idx]]]; // Get probability of correct token
            loss -= prob_target.ln();
        }

        loss / target.len() as f32
    }

    fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone();

        if probs.shape()[0] != target.len() {
            panic!("Probs and target must have the same number of rows");
        }
        
        // Process each row in the probability matrix
        for row_idx in 0..grads.shape()[0] {
            grads[[row_idx, target[row_idx]]] -= 1.0; // Subtract 1.0 from the target column in each row
        }
        
        grads
    }
}

pub struct LayerNorm {
    epsilon: f32,   // Small constant for stability
    gamma: Array2<f32>, // Learnable scaling parameter
    beta: Array2<f32>,  // Learnable bias parameter
}

impl LayerNorm {
    /// Initialize LayerNorm with learnable parameters
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            beta: Array2::zeros((1, embedding_dim)), // Initialize beta to 0
        }
    }

    pub fn normalize(&self, input: &Array2<f32>) -> Array2<f32> {
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // Mean per token
        let std = input.std_axis(Axis(1), 0.0).insert_axis(Axis(1)); // Std per token

        // Normalize: (X - mean) / (std + epsilon)
        let normalized = (input - &mean) / (&std + self.epsilon);
        normalized * &self.gamma + &self.beta // Apply gamma & beta
    }
}