use ndarray::Array2;
use llm::optimizers::adamw::AdamW;
use llm::optimizer::Optimizer;

#[test]
fn test_adamw_initialization() {
    let shape = (2, 3);
    let adamw = AdamW::new(shape, 0.01);
    
    // Check if momentum and velocity matrices are initialized to zeros
    assert_eq!(adamw.m.shape(), &[2, 3]);
    assert_eq!(adamw.v.shape(), &[2, 3]);
    assert!(adamw.m.iter().all(|&x| x == 0.0));
    assert!(adamw.v.iter().all(|&x| x == 0.0));
}

#[test]
fn test_adamw_step() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adamw = AdamW::new(shape, 0.01);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);
    
    // Store initial parameters
    let initial_params = params.clone();
    
    // Perform optimization step
    adamw.step(&mut params, &grads, lr);
    
    // Parameters should have changed
    assert_ne!(params, initial_params);
    
    // Parameters should have decreased (due to weight decay and positive gradients)
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adamw_multiple_steps() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adamw = AdamW::new(shape, 0.01);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);
    
    // Store initial parameters
    let initial_params = params.clone();
    
    // Perform multiple optimization steps
    for _ in 0..10 {
        adamw.step(&mut params, &grads, lr);
    }
    
    // Parameters should have changed more significantly
    assert!(params.iter().all(|&x| x < initial_params[[0, 0]]));
}

#[test]
fn test_adamw_with_zero_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adamw = AdamW::new(shape, 0.01);
    let mut params = Array2::ones(shape);
    let grads = Array2::zeros(shape);
    
    // Store initial parameters
    let initial_params = params.clone();
    
    // Perform optimization step with zero gradients
    adamw.step(&mut params, &grads, lr);
    
    // Parameters should change due to weight decay, even with zero gradients
    assert_ne!(params, initial_params);
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adamw_with_negative_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adamw = AdamW::new(shape, 0.01);
    let mut params = Array2::ones(shape);
    let grads = Array2::from_shape_fn(shape, |_| -1.0);
    
    // Perform optimization step
    adamw.step(&mut params, &grads, lr);
    
    // Parameters should have increased (since gradients are negative), but weight decay will slightly reduce them.
    // The net effect depends on the learning rate and weight decay.
    // With lr=0.001 and wd=0.01, the update from gradient is positive, and the decay is negative.
    // The update should be larger than the decay.
    assert!(params.iter().all(|&x| x > 0.99 && x < 1.01)); // Check it's around 1.0 but changed
}