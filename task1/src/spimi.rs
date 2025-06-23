use crate::{Dictionary, TermEntry};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// SPIMI (Single-Pass In-Memory Indexing) implementation
/// Builds inverted indexes efficiently for large document collections
pub struct SPIMIIndexer {
    memory_limit: usize,
    output_dir: String,
    current_memory_usage: usize,
    current_index: HashMap<String, Vec<String>>,
    block_count: usize,
}

impl SPIMIIndexer {
    pub fn new<P: AsRef<Path>>(memory_limit_mb: usize, output_dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let output_path = output_dir.as_ref();
        if !output_path.exists() {
            fs::create_dir_all(output_path)?;
        }

        Ok(SPIMIIndexer {
            memory_limit: memory_limit_mb * 1024 * 1024, // Convert MB to bytes
            output_dir: output_path.to_string_lossy().to_string(),
            current_memory_usage: 0,
            current_index: HashMap::new(),
            block_count: 0,
        })
    }

    pub fn add_document(&mut self, doc_id: &str, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        let words = self.tokenize(text);

        for word in words {
            let term_size = word.len() + doc_id.len() + 32; // Estimate memory usage

            if self.current_memory_usage + term_size > self.memory_limit {
                self.write_block_to_disk()?;
                self.reset_current_block();
            }

            self.current_index
                .entry(word)
                .or_insert_with(Vec::new)
                .push(doc_id.to_string());

            self.current_memory_usage += term_size;
        }

        Ok(())
    }

    pub fn finalize(&mut self) -> Result<Dictionary, Box<dyn std::error::Error>> {
        // Write the last block
        if !self.current_index.is_empty() {
            self.write_block_to_disk()?;
        }

        println!("SPIMI: Merging {} blocks into final index", self.block_count);
        self.merge_blocks()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty() && word.len() > 2)
            .collect()
    }

    fn write_block_to_disk(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let block_path = format!("{}/block_{}.txt", self.output_dir, self.block_count);
        let file = File::create(&block_path)?;
        let mut writer = BufWriter::new(file);

        // Sort terms for better merging performance
        let mut terms: Vec<_> = self.current_index.iter().collect();
        terms.sort_by_key(|(term, _)| term.as_str());

        println!("SPIMI: Writing block {} with {} terms ({:.2} MB)",
                 self.block_count, terms.len(),
                 self.current_memory_usage as f64 / 1024.0 / 1024.0);

        for (term, docs) in terms {
            // Deduplicate and sort document IDs
            let mut unique_docs = docs.clone();
            unique_docs.sort();
            unique_docs.dedup();

            writeln!(writer, "{}:{}", term, unique_docs.join(","))?;
        }

        writer.flush()?;
        self.block_count += 1;
        Ok(())
    }

    fn reset_current_block(&mut self) {
        self.current_index.clear();
        self.current_memory_usage = 0;
    }

    fn merge_blocks(&self) -> Result<Dictionary, Box<dyn std::error::Error>> {
        let mut dictionary = Dictionary::new();
        let mut block_readers = Vec::new();
        let mut current_lines = Vec::new();

        // Open all block files
        for i in 0..self.block_count {
            let block_path = format!("{}/block_{}.txt", self.output_dir, i);
            let file = File::open(&block_path)?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            if let Some(Ok(first_line)) = lines.next() {
                current_lines.push(Some(first_line));
                block_readers.push(lines);
            } else {
                current_lines.push(None);
                block_readers.push(lines);
            }
        }

        let mut merged_terms = 0;

        loop {
            // Find the lexicographically smallest term among all blocks
            let mut min_term: Option<String> = None;
            let mut min_indices = Vec::new();

            for (i, line_opt) in current_lines.iter().enumerate() {
                if let Some(line) = line_opt {
                    if let Some(term) = line.split(':').next() {
                        match &min_term {
                            None => {
                                min_term = Some(term.to_string());
                                min_indices = vec![i];
                            }
                            Some(current_min) if term < current_min.as_str() => {
                                min_term = Some(term.to_string());
                                min_indices = vec![i];
                            }
                            Some(current_min) if term == current_min.as_str() => {
                                min_indices.push(i);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if min_term.is_none() {
                break; // No more terms to process
            }

            let term = min_term.unwrap();
            let mut all_docs = Vec::new();

            // Collect all documents for this term from relevant blocks
            for &block_idx in &min_indices {
                if let Some(line) = &current_lines[block_idx] {
                    if let Some(docs_part) = line.split(':').nth(1) {
                        for doc in docs_part.split(',') {
                            if !doc.is_empty() {
                                all_docs.push(doc.to_string());
                            }
                        }
                    }
                }

                // Advance to next line in this block
                current_lines[block_idx] = block_readers[block_idx].next().transpose()?;
            }

            // Deduplicate and create dictionary entry
            all_docs.sort();
            all_docs.dedup();

            dictionary.terms.insert(term.clone(), TermEntry {
                frequency: all_docs.len() as u32,
                documents: all_docs,
            });

            merged_terms += 1;
            if merged_terms % 10000 == 0 {
                println!("SPIMI: Merged {} terms", merged_terms);
            }
        }

        println!("SPIMI: Merge complete - {} unique terms", merged_terms);

        // Clean up temporary files
        for i in 0..self.block_count {
            let block_path = format!("{}/block_{}.txt", self.output_dir, i);
            let _ = fs::remove_file(block_path); // Ignore errors
        }

        Ok(dictionary)
    }

    pub fn memory_usage(&self) -> usize {
        self.current_memory_usage
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// Parallel SPIMI implementation for better performance
pub struct ParallelSPIMIIndexer {
    memory_limit_per_thread: usize,
    output_dir: String,
    num_threads: usize,
}

impl ParallelSPIMIIndexer {
    pub fn new<P: AsRef<Path>>(
        memory_limit_mb: usize,
        output_dir: P,
        num_threads: Option<usize>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_path = output_dir.as_ref();
        if !output_path.exists() {
            fs::create_dir_all(output_path)?;
        }

        let threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Ok(ParallelSPIMIIndexer {
            memory_limit_per_thread: memory_limit_mb * 1024 * 1024 / threads,
            output_dir: output_path.to_string_lossy().to_string(),
            num_threads: threads,
        })
    }

    pub fn build_index<F>(&self, documents: Vec<(String, String)>,
                         progress_callback: F) -> Result<Dictionary, Box<dyn std::error::Error>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let chunk_size = (documents.len() + self.num_threads - 1) / self.num_threads;
        let chunks: Vec<_> = documents.chunks(chunk_size).enumerate().collect();

        println!("Parallel SPIMI: Processing {} documents across {} threads",
                 documents.len(), self.num_threads);

        // Process chunks sequentially to avoid error propagation issues
        let mut partial_dictionaries = Vec::new();

        for (chunk_idx, chunk) in chunks {
            let thread_output_dir = format!("{}/thread_{}", self.output_dir, chunk_idx);
            fs::create_dir_all(&thread_output_dir)?;

            let mut indexer = SPIMIIndexer::new(
                self.memory_limit_per_thread / (1024 * 1024),
                &thread_output_dir
            )?;

            for (i, (doc_id, text)) in chunk.iter().enumerate() {
                indexer.add_document(doc_id, text)?;

                if i % 1000 == 0 {
                    progress_callback(chunk_idx * chunk_size + i, documents.len());
                }
            }

            partial_dictionaries.push(indexer.finalize()?);
        }

        println!("Parallel SPIMI: Merging {} partial dictionaries", partial_dictionaries.len());
        self.merge_dictionaries(partial_dictionaries)
    }

    fn merge_dictionaries(&self, dictionaries: Vec<Dictionary>) -> Result<Dictionary, Box<dyn std::error::Error>> {
        let mut final_dict = Dictionary::new();

        for dict in dictionaries {
            for (term, entry) in dict.terms {
                let final_entry = final_dict.terms.entry(term).or_insert_with(|| TermEntry {
                    frequency: 0,
                    documents: Vec::new(),
                });

                final_entry.frequency += entry.frequency;
                final_entry.documents.extend(entry.documents);
            }

            final_dict.total_words += dict.total_words;
            final_dict.total_documents += dict.total_documents;
            final_dict.collection_size_bytes += dict.collection_size_bytes;
        }

        // Deduplicate documents in final merge
        for entry in final_dict.terms.values_mut() {
            entry.documents.sort();
            entry.documents.dedup();
            entry.frequency = entry.documents.len() as u32;
        }

        println!("Parallel SPIMI: Final merge complete - {} unique terms", final_dict.terms.len());
        Ok(final_dict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_spimi_indexer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let indexer = SPIMIIndexer::new(1, temp_dir.path()).unwrap();
        assert_eq!(indexer.memory_limit, 1024 * 1024);
        assert_eq!(indexer.block_count, 0);
    }

    #[test]
    fn test_tokenization() {
        let temp_dir = TempDir::new().unwrap();
        let indexer = SPIMIIndexer::new(1, temp_dir.path()).unwrap();
        let tokens = indexer.tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }
}
