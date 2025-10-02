use llm::adam::Adam;
use ndarray::Array2;

#[test]
fn test_adam_initialization() {
    let shape = [2, 3];
    let adam = Adam::new((2, 3));

    // Check if momentum and velocity matrices are initialized to zeros
    assert_eq!(adam.m.shape(), shape);
    assert_eq!(adam.v.shape(), shape);
    assert!(adam.m.iter().all(|&x| x == 0.0));
    assert!(adam.v.iter().all(|&x| x == 0.0));
}

#[test]
fn test_adam_step() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have changed
    assert_ne!(params, initial_params);

    // Parameters should have decreased (since gradients are positive)
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adam_multiple_steps() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform multiple optimization steps
    for _ in 0..10 {
        adam.step(&mut params, &grads, lr);
    }

    // Parameters should have changed more significantly
    assert!(params.iter().all(|&x| x < initial_params[[0, 0]]));
}

#[test]
fn test_adam_with_zero_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::zeros(shape);

    // Store initial parameters
    let initial_params = params.clone();

    // Perform optimization step with zero gradients
    adam.step(&mut params, &grads, lr);

    // Parameters should not change with zero gradients
    assert_eq!(params, initial_params);
}

#[test]
fn test_adam_with_negative_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::from_shape_fn(shape, |_| -1.0);

    // Perform optimization step
    adam.step(&mut params, &grads, lr);

    // Parameters should have increased (since gradients are negative)
    assert!(params.iter().all(|&x| x > 1.0));
}
