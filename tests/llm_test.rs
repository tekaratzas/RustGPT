use llm::{LLM, Vocab, Layer};
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
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cache_input = Some(input.clone());
        let mut mock_output = Array2::zeros((1, self.vocab_size));

        // Force stop after 5 loops to match expected output
        if self.loop_count >= self.stop_loop_count {
            mock_output[[0, self.stop_index]] = 1.0;
        } else {
            mock_output[[0, 0]] = 1.0;
        }

        self.loop_count += 1;
        mock_output
    }

    // Need to test this next
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // println!("Grads: {:?}", grads);
        let input = self.cache_input.as_ref().unwrap();

        // use chain rule
        let grad_input = input.t().dot(grads);
        self.cached_grads = Some(grad_input.clone());

        // println!("Grad input: {:?}", grad_input);
        return grad_input
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
    let llm = LLM::new(vocab, vec![
        Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))
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
        Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))
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

#[test]
fn test_llm_train() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let layer = Box::new(TestOutputProjectionLayer::new(5, 1, vocab_size));
    let mut llm = LLM::new(vocab.clone(), vec![
        layer
    ]);

    let training_data = vec![
        ("hello world this is </s>", "rust </s>"),
    ];

    llm.train(training_data, 10, 0.01);
}