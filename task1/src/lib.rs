pub mod bigram_index;
pub mod coordinate_index;
pub mod dictionary;
pub mod incidence_matrix;
pub mod inverted_index;
pub mod parser;
pub mod permutation_index;
pub mod query;
pub mod suffix_tree;
pub mod trigram_index;
pub mod wildcard_search;

pub use bigram_index::*;
pub use coordinate_index::*;
pub use dictionary::*;
pub use incidence_matrix::*;
pub use inverted_index::*;
pub use parser::*;
pub use permutation_index::*;
pub use query::*;
pub use suffix_tree::*;
pub use trigram_index::*;
pub use wildcard_search::*;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

pub fn collect_fb2_files(directory: &str) -> Vec<std::path::PathBuf> {
    WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "fb2"))
        .map(|e| e.path().to_path_buf())
        .collect()
}

pub fn build_dictionary(
    files: &[std::path::PathBuf],
    show_progress: bool,
) -> Result<Dictionary, Box<dyn std::error::Error>> {
    let mut dictionary = Dictionary::new();

    let pb = if show_progress {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap(),
        );
        Arc::new(Mutex::new(Some(pb)))
    } else {
        Arc::new(Mutex::new(None))
    };

    let pb_clone = Arc::clone(&pb);

    println!("Processing {} files with parallel parser...", files.len());

    // Process files in parallel and collect results
    let results: Vec<_> = files
        .par_iter()
        .enumerate()
        .map(|(index, file_path)| {
            let parser = FB2Parser::new();

            // Update progress bar
            if let Ok(pb_lock) = pb_clone.lock() {
                if let Some(ref pb) = *pb_lock {
                    pb.set_message(format!("Processing {}", file_path.display()));
                }
            }

            if index < 5 || index % 50 == 0 {
                println!(
                    "  Processing file {}/{}: {}",
                    index + 1,
                    files.len(),
                    file_path.display()
                );
            }

            let file_size = match fs::metadata(file_path) {
                Ok(metadata) => metadata.len(),
                Err(e) => {
                    eprintln!("Error reading metadata for {}: {}", file_path.display(), e);
                    return None;
                }
            };

            if file_size < 150_000 {
                if index < 5 {
                    eprintln!(
                        "Warning: {} is smaller than 150KB ({} bytes)",
                        file_path.display(),
                        file_size
                    );
                }
                return None;
            }

            let document_name = file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let words = match parser.parse_file(file_path) {
                Ok(words) => {
                    if index < 5 {
                        println!("    Parsed {} words from {}", words.len(), document_name);
                    }
                    words
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", file_path.display(), e);
                    return None;
                }
            };

            // Update progress bar
            if let Ok(pb_lock) = pb_clone.lock() {
                if let Some(ref pb) = *pb_lock {
                    pb.inc(1);
                }
            }

            Some((file_size, document_name, words))
        })
        .collect();

    println!(
        "Parallel processing complete, {} results collected",
        results.len()
    );

    // Merge results into dictionary sequentially
    println!("Merging results into dictionary...");
    let mut merged_count = 0;
    for result in results {
        if let Some((file_size, document_name, words)) = result {
            merged_count += 1;
            if merged_count <= 5 || merged_count % 50 == 0 {
                println!(
                    "  Merging document {}: {} ({} words)",
                    merged_count,
                    document_name,
                    words.len()
                );
            }

            dictionary.add_file_stats(file_size);

            let terms: Vec<(String, String)> = words
                .into_iter()
                .map(|word| (word, document_name.clone()))
                .collect();

            dictionary.merge_terms(terms);

            if merged_count <= 5 || merged_count % 50 == 0 {
                println!(
                    "    Dictionary now has {} unique terms",
                    dictionary.terms.len()
                );
            }
        }
    }
    println!(
        "Dictionary merge complete - {} documents processed",
        merged_count
    );

    if let Ok(pb_lock) = pb.lock() {
        if let Some(ref pb) = *pb_lock {
            pb.finish_with_message("Dictionary building completed");
        }
    }

    Ok(dictionary)
}
