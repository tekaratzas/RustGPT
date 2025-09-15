// Tests for the Dataset struct in dataset_loader.rs

use llm::Dataset;

#[test]
fn test_dataset_new_initializes_data() {
    let dataset = Dataset::new();
    // Check pretraining data
    assert!(!dataset.pretraining_data.is_empty(), "Pretraining data should not be empty");
    assert_eq!(dataset.pretraining_data.len(), dataset.raw_pretraining_data.len());
    // Check chat training data
    assert!(!dataset.chat_training_data.is_empty(), "Chat training data should not be empty");
    assert_eq!(dataset.chat_training_data.len(), dataset.raw_chat_training_data.len());
    // Check that the first pretraining example is as expected
    assert_eq!(dataset.pretraining_data[0], "The sun rises in the east and sets in the west </s>");
    // Check that the first chat training example is as expected
    assert!(dataset.chat_training_data[0].starts_with("User: What causes rain?"));
}
