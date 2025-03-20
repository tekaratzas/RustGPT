use llm::{LLM, Vocab, Layer};
use ndarray::{array, Array2};

struct TestOutputProjectionLayer {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub cache_weights: Option<Array2<f32>>,
    pub cache_bias: Option<Array2<f32>>,
    pub loop_count: usize,
}

impl Layer for TestOutputProjectionLayer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let input_width = input.shape()[1];
        let mut mock_output = Array2::zeros((1, input_width));

        // Force stop after 5 loops to match expected output
        if self.loop_count >= 5 {
            mock_output[[0, 5]] = 1.0;
        } else if input_width > 0 {
            mock_output[[0, 0]] = 1.0;
        }

        self.loop_count += 1;
        mock_output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        Array2::zeros(self.weights.dim())
    }
}

impl TestOutputProjectionLayer {
    pub fn new(vocab_size: usize) -> Self {
        TestOutputProjectionLayer {
            weights: Array2::zeros((2, vocab_size)),
            bias: Array2::zeros((1, vocab_size)),
            cache_weights: None,
            cache_bias: None,
            loop_count: 0,
        }
    }
}

#[test]
fn test_llm_tokenize() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let llm = LLM::new(vocab, vec![
        Box::new(TestOutputProjectionLayer::new(vocab_size))
    ]);
    
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
    let mut llm = LLM::new(vocab.clone(), vec![
        Box::new(TestOutputProjectionLayer::new(vocab_size))
    ]);
    
    // Test prediction
    let input_text = "hello world this is rust";
    let input_tokens = llm.tokenize(input_text);
    let result = llm.predict(input_text);
    assert!(!result.is_empty());

    // Build expected output
    let mut expected_tokens = vec![0; input_tokens.len()].iter().map(|x| vocab.decode[x].clone()).collect::<Vec<String>>();
    expected_tokens.push("</s>".to_string());
    let expected_output = expected_tokens.join(" ");

    assert_eq!(result, expected_output);    
} 