use ndarray::{s, Array2};
use rand::prelude::*;
use crate::{vocab::Vocab, llm::Layer, EMBEDDING_DIM, MAX_SEQ_LEN};

pub struct Embeddings {
    pub token_embeddings: Array2<f32>,
    pub positional_embeddings: Array2<f32>,
}

impl Default for Embeddings { 
    fn default() -> Self {
        Self { 
            token_embeddings: Self::init_embeddings(Vocab::default_words().len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
         }
    }
}

impl Embeddings {

    pub fn new(vocab: Vocab) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| rng.random_range(-1.0..1.0))
    }

    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| rng.random_range(-1.0..1.0))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    fn get_positional_embeddings(positional_encodings: &Array2<f32>, seq_len: usize) -> Array2<f32> {
        positional_encodings.slice(s![0..seq_len, ..]).to_owned()
    }

    pub fn embed_tokens(
        &self,
        token_ids: &[usize]
    ) -> Array2<f32> {
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        let position_embeds = Self::get_positional_embeddings(&self.positional_embeddings, token_ids.len());
        token_embeds + position_embeds // Element-wise sum
    }
}

impl Layer for Embeddings {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        println!("input: {:?}", input);
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        println!("token_ids: {:?}", token_ids);
        self.embed_tokens(&token_ids)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_token_embeddings = grads.dot(&self.positional_embeddings.t());
        let grad_positional_embeddings = grads.t().dot(&self.token_embeddings);
        grad_token_embeddings + grad_positional_embeddings
    }
}   
