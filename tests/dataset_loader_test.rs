// Tests for the Dataset struct in dataset_loader.rs

use llm::{Dataset, DatasetType};


#[test]
fn test_dataset_new_json() {
    let dataset = Dataset::new(
        "data/pretraining_data.json".to_string(),
        "data/chat_training_data.json".to_string(),
        DatasetType::JSON,
    );
    assert!(!dataset.pretraining_data.is_empty(), "Pretraining data should not be empty");
    assert!(!dataset.chat_training_data.is_empty(), "Chat training data should not be empty");
    assert_eq!(dataset.pretraining_data[0], "The sun rises in the east and sets in the west </s>");
    assert!(dataset.chat_training_data[0].starts_with("User: What causes rain?"));
}

#[test]
fn test_dataset_new_csv() {
    // Prepare test CSV files with minimal data
    let pretraining_csv = "data/pretraining_data_test.csv";
    let chat_csv = "data/chat_training_data_test.csv";
    std::fs::write(pretraining_csv, "The sun rises in the east and sets in the west </s>\nWater flows downhill due to gravity </s>").unwrap();
    std::fs::write(chat_csv, "User: What causes rain?\nUser: How do mountains form?").unwrap();

    let dataset = Dataset::new(
        pretraining_csv.to_string(),
        chat_csv.to_string(),
        DatasetType::CSV,
    );
    assert_eq!(dataset.pretraining_data[0], "The sun rises in the east and sets in the west </s>");
    assert_eq!(dataset.pretraining_data[1], "Water flows downhill due to gravity </s>");
    assert_eq!(dataset.chat_training_data[0], "User: What causes rain?");
    assert_eq!(dataset.chat_training_data[1], "User: How do mountains form?");

    // Clean up test files
    std::fs::remove_file(pretraining_csv).unwrap();
    std::fs::remove_file(chat_csv).unwrap();
}
