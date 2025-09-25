use llm::transformer::TransformerBlock;
use llm::EMBEDDING_DIM;
use llm::Embeddings;
use llm::output_projection::OutputProjection;
use llm::HIDDEN_DIM;
use llm::MAX_SEQ_LEN;
use llm::{LLM, Layer, Vocab};
use ndarray::Array2;

struct TestOutputProjectionLayer {
    pub cache_input: Option<Array2<f32>>,
    pub loop_count: usize,
    pub stop_index: usize,
    pub stop_loop_count: usize,
    pub vocab_size: usize,
    pub cached_grads: Option<Array2<f32>>,
}

impl Layer for TestOutputProjectionLayer {
    fn layer_type(&self) -> &str {
        "TestOutputProjectionLayer"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cache_input = Some(input.clone());
        let mut mock_output = Array2::zeros((input.shape()[1], self.vocab_size));

        let last_token_index = input.shape()[1] - 1;

        // Force stop after 5 loops to match expected output
        if self.loop_count >= self.stop_loop_count {
            mock_output[[last_token_index, self.stop_index]] = 1.0;
        } else {
            mock_output[[last_token_index, 0]] = 1.0;
        }

        self.loop_count += 1;
        mock_output
    }

    // Need to test this next
    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        let input = self.cache_input.as_ref().unwrap();

        // use chain rule
        let grad_input = input.dot(grads);
        self.cached_grads = Some(grad_input.clone());

        return grad_input;
    }
}

impl TestOutputProjectionLayer {
    pub fn new(stop_index: usize, stop_loop_count: usize, vocab_size: usize) -> Self {
        TestOutputProjectionLayer {
            cache_input: None,
            loop_count: 0,
            stop_index,
            stop_loop_count,
            vocab_size,
            cached_grads: None,
        }
    }
}

#[test]
fn test_llm_tokenize() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let llm = LLM::new(
        vocab,
        vec![Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))],
    );

    // Test tokenization
    let tokens = llm.tokenize("hello world");
    assert!(!tokens.is_empty());

    // Test that tokens can be decoded back
    for token in tokens {
        assert!(llm.vocab.decode(token).is_some());
    }
}

#[test]
fn test_llm_predict() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let mut llm = LLM::new(
        vocab.clone(),
        vec![Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))],
    );

    // Test prediction
    let input_text = "hello world this is rust";
    let input_tokens = llm.tokenize(input_text);
    let result = llm.predict(input_text);
    assert!(!result.is_empty());

    // Build expected output
    let mut expected_tokens = vec![0; input_tokens.len()]
        .iter()
        .map(|x| vocab.decode[x].clone())
        .collect::<Vec<String>>();
    expected_tokens.push("</s>".to_string());
    let expected_output = expected_tokens.join(" ");

    assert_eq!(result, expected_output);
}

#[test]
fn test_llm_train() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let layer = Box::new(TestOutputProjectionLayer::new(5, 1, vocab_size));
    let mut llm = LLM::new(vocab.clone(), vec![layer]);

    let training_data = vec!["hello world this is rust."];

    llm.train(training_data, 10, 0.01);
}

#[test]
fn test_llm_integration() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();

    let embeddings = Box::new(Embeddings::new(vocab.clone()));
    let output_projection = Box::new(OutputProjection::new(EMBEDDING_DIM, vocab_size));

    let mut llm = LLM::new(vocab.clone(), vec![embeddings, output_projection]);

    let input_text = "hello world this is rust";
    llm.train(vec![input_text], 10, 0.01);
}

#[test]
fn test_llm_total_parameters() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    
    // Create an LLM with actual layers to get a meaningful parameter count
    let embeddings = Box::new(Embeddings::new(vocab.clone()));
    let transformer_block = Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM));
    let output_projection = Box::new(OutputProjection::new(EMBEDDING_DIM, vocab_size));
    
    let llm = LLM::new(vocab.clone(), vec![embeddings, transformer_block, output_projection]);
    
    // The total parameters should be greater than 0 for a model with actual layers
    let param_count = llm.total_parameters();
    assert!(param_count > 0);

    // Let's validate that this is equal to the expected total number of parameters. (based on our source)
    let expected_embeddings_parameters = vocab_size * EMBEDDING_DIM + MAX_SEQ_LEN * EMBEDDING_DIM;
    let expected_transformer_block_parameters = 
    (2 * EMBEDDING_DIM) + // LayerNorm
    (3 * EMBEDDING_DIM * EMBEDDING_DIM) + // SelfAttention
    (2 * EMBEDDING_DIM) + // LayerNorm
    (EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM); // FeedForward
    let expected_output_projection_parameters = EMBEDDING_DIM * vocab_size + vocab_size;
    assert!(param_count == expected_embeddings_parameters + 
        expected_transformer_block_parameters + expected_output_projection_parameters);
}