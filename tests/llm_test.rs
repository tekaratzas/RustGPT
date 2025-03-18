use llm::{LLM, Vocab, Layer};
use ndarray::{array, Array2};

struct TestLayer {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub cache_weights: Option<Array2<f32>>,
    pub cache_bias: Option<Array2<f32>>,
}

impl Layer for TestLayer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let input_width = input.shape()[1];
        let mock_output = Array2::ones((1, input_width));

        mock_output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        Array2::zeros(self.weights.dim())
    }
}

impl TestLayer {
    pub fn new() -> Self {
        TestLayer {
            weights: Array2::zeros((1, 1)),
            bias: Array2::zeros((1, 1)),
            cache_weights: None,
            cache_bias: None,
        }
    }
}

#[test]
fn test_llm_tokenize() {
    let vocab = Vocab::default();
    let llm = LLM::new(vocab, vec![
        Box::new(TestLayer::new())
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
    let mut llm = LLM::new(vocab.clone(), vec![
        Box::new(TestLayer::new())
    ]);
    
    // Test prediction
    let result = llm.predict("hello world");
    assert!(!result.is_empty());

    let expected_tokens = vec![1, 1, 1, 1].iter().map(|x| vocab.decode[x].clone()).collect::<Vec<String>>();
    let expected_output = expected_tokens.join(" ");
    assert_eq!(result, expected_output);    
} 