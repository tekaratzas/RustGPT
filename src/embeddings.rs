use ndarray::{s, Array2};
use rand_distr::{Normal, Distribution};
use crate::{vocab::Vocab, llm::Layer, EMBEDDING_DIM, MAX_SEQ_LEN, adam::Adam};

pub struct Embeddings {
    pub token_embeddings: Array2<f32>,
    pub positional_embeddings: Array2<f32>,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
    pub positional_optimizer: Adam,
}

impl Default for Embeddings { 
    fn default() -> Self {
        Self { 
            token_embeddings: Self::init_embeddings(Vocab::default_words().len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((Vocab::default_words().len(), EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM))
        }
    }
}

impl Embeddings {

    pub fn new(vocab: Vocab) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap(); // Increased for better learning
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap(); // Increased for better learning
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= embeddings.nrows() {
                panic!("Token ID {} out of bounds for vocab size {}", token_id, embeddings.nrows());
            }
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    fn get_positional_embeddings(positional_encodings: &Array2<f32>, seq_len: usize) -> Array2<f32> {
        if seq_len > positional_encodings.nrows() {
            panic!("Sequence length {} exceeds maximum {}", seq_len, positional_encodings.nrows());
        }
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
    // fn input_shape(&self) -> &[usize] {
    //     &[MAX_SEQ_LEN]
    // }

    // fn output_shape(&self) -> &[usize] {
    //     &[MAX_SEQ_LEN, EMBEDDING_DIM]
    // }

    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> { // input shape is [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&token_ids) // shape is [sequence_length, embedding_dim]
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for embeddings
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());
        let mut positional_grads = Array2::zeros(self.positional_embeddings.dim());

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.token_embeddings.nrows() {
                panic!("Token ID {} out of bounds for vocab size {}", token_id, self.token_embeddings.nrows());
            }
            let grad_row = grads.row(i);
            
            // Accumulate token embedding gradients efficiently (no temp variable)
            {
                let mut token_row = token_grads.row_mut(token_id);
                token_row += &grad_row;
            }
            
            // Accumulate positional embedding gradients efficiently (no temp variable)
            {
                let mut pos_row = positional_grads.row_mut(i);
                pos_row += &grad_row;
            }
        }

        self.token_optimizer.step(&mut self.token_embeddings, &token_grads, lr);
        self.positional_optimizer.step(&mut self.positional_embeddings, &positional_grads, lr);

        // Return gradient to propagate further back
        grads.to_owned()
    }
}
