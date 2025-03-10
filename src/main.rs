use embeddings::Embeddings;
use llm::LLM;
use vocab::Vocab;

mod llm;
mod embeddings;
mod vocab;
mod transformer;
mod feed_forward;
mod self_attention;
mod output_projection;
// Constants
const MAX_SEQ_LEN: usize = 10;
const EMBEDDING_DIM: usize = 4;
const HIDDEN_DIM: usize = 4;

fn main() {
    // Mock input
    let string = String::from("hello world </s>");

    let llm = LLM::default();

    let result = llm.predict(&string);
    println!("output of LLM: {}", result);
}
