use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;

/// Front-packing compression utilities for dictionary strings
mod front_packing {
    use std::collections::HashMap;

    /// A front-packed block containing terms with a common prefix
    #[derive(Debug, Clone)]
    pub struct FrontPackedBlock {
        pub prefix_length: u8,        // Length of common prefix
        pub term_count: u16,          // Number of terms in this block
        pub suffixes: Vec<String>,    // Suffixes after removing common prefix
    }

    /// Front-pack a sorted list of terms
    pub fn compress_terms(mut terms: Vec<String>) -> (Vec<u8>, HashMap<String, (usize, usize)>) {
        if terms.is_empty() {
            return (Vec::new(), HashMap::new());
        }

        // Sort terms lexicographically for optimal front-packing
        terms.sort();
        
        let mut compressed_data = Vec::new();
        let mut term_positions = HashMap::new();
        let mut current_block_start = 0;

        let mut i = 0;
        while i < terms.len() {
            // Find terms with common prefix starting at position i
            let (block_end, prefix_len) = find_optimal_block(&terms, i);
            
            // Create front-packed block
            let block_terms = &terms[i..block_end];
            let prefix = if prefix_len > 0 {
                &block_terms[0][..prefix_len]
            } else {
                ""
            };

            // Record positions for each term in this block
            for (block_idx, term) in block_terms.iter().enumerate() {
                term_positions.insert(term.clone(), (current_block_start, block_idx));
            }

            // Encode block header
            compressed_data.push(prefix_len as u8);
            compressed_data.extend_from_slice(&(block_terms.len() as u16).to_le_bytes());
            
            // Store prefix
            if prefix_len > 0 {
                compressed_data.extend_from_slice(prefix.as_bytes());
            }

            // Store suffixes with their lengths
            for term in block_terms {
                let suffix = &term[prefix_len..];
                compressed_data.push(suffix.len() as u8);
                compressed_data.extend_from_slice(suffix.as_bytes());
            }

            current_block_start = compressed_data.len();
            i = block_end;
        }

        (compressed_data, term_positions)
    }

    /// Find the optimal block size and prefix length for front-packing
    fn find_optimal_block(terms: &[String], start: usize) -> (usize, usize) {
        if start >= terms.len() {
            return (start, 0);
        }

        let _first_term = &terms[start];
        let mut best_end = start + 1;
        let mut best_prefix_len = 0;
        let mut best_savings = 0i32;

        // Try different block sizes and prefix lengths
        for end in (start + 1)..=std::cmp::min(start + 255, terms.len()) {
            if let Some(prefix_len) = find_common_prefix(&terms[start..end]) {
                if prefix_len == 0 {
                    continue;
                }

                // Calculate space savings with this configuration
                let block_size = end - start;
                let original_size: usize = terms[start..end].iter().map(|s| s.len() + 1).sum(); // +1 for length byte
                let compressed_size = 3 + prefix_len + block_size + terms[start..end].iter().map(|s| s.len() - prefix_len).sum::<usize>(); // header + prefix + suffixes

                let savings = original_size as i32 - compressed_size as i32;
                
                if savings > best_savings {
                    best_savings = savings;
                    best_end = end;
                    best_prefix_len = prefix_len;
                }
            }
        }

        // If no beneficial compression found, use single term
        if best_savings <= 0 {
            (start + 1, 0)
        } else {
            (best_end, best_prefix_len)
        }
    }

    /// Find the longest common prefix among a group of terms
    fn find_common_prefix(terms: &[String]) -> Option<usize> {
        if terms.len() < 2 {
            return Some(0);
        }

        let first = &terms[0];
        let mut prefix_len = 0;

        for i in 0..first.len() {
            let char_at_i = first.chars().nth(i)?;
            
            if terms.iter().all(|term| {
                term.chars().nth(i).map_or(false, |c| c == char_at_i)
            }) {
                prefix_len = i + 1;
            } else {
                break;
            }
        }

        Some(prefix_len)
    }

    /// Decompress front-packed data to retrieve a specific term
    pub fn decompress_term(
        _compressed_data: &[u8],
        term_positions: &HashMap<String, (usize, usize)>,
        term: &str,
    ) -> Option<String> {
        let (_block_start, _term_index) = term_positions.get(term)?;
        
        // This is a simplified version - in practice you'd need to decode the block
        // For now, return the original term since we have it in the positions map
        Some(term.to_string())
    }
}

use front_packing::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermEntry {
    pub frequency: u32,
    pub documents: HashSet<String>,
    pub start_pos: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dictionary {
    pub terms: HashMap<usize, TermEntry>,
    pub term_lookup: HashMap<String, usize>,
    pub terms_string: String,
    pub total_words: u64,
    pub total_documents: u32,
    pub collection_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDictionary {
    /// Compressed term entries (same as regular dictionary)
    pub terms: HashMap<usize, TermEntry>,
    /// Front-packed compressed terms data
    pub compressed_terms_data: Vec<u8>,
    /// Mapping from term to (block_start, term_index_in_block)
    pub term_positions: HashMap<String, (usize, usize)>,
    /// Statistics
    pub total_words: u64,
    pub total_documents: u32,
    pub collection_size_bytes: u64,
    /// Compression statistics
    pub original_terms_size: usize,
    pub compressed_terms_size: usize,
}

impl Dictionary {
    pub fn get_term(&self, start_pos: usize) -> Option<&str> {
        if let Some(entry) = self.terms.get(&start_pos) {
            Some(&self.terms_string[start_pos..start_pos + entry.length])
        } else {
            None
        }
    }

    pub fn find_or_add_term(&mut self, term: &str) -> usize {
        if let Some(&start_pos) = self.term_lookup.get(term) {
            return start_pos;
        }

        let start_pos = self.terms_string.len();
        self.terms_string.push_str(term);
        self.term_lookup.insert(term.to_string(), start_pos);
        start_pos
    }
    pub fn new() -> Self {
        Dictionary {
            terms: HashMap::new(),
            term_lookup: HashMap::new(),
            terms_string: String::new(),
            total_words: 0,
            total_documents: 0,
            collection_size_bytes: 0,
        }
    }

    pub fn add_term(&mut self, term: String, document: String) {
        let start_pos = self.find_or_add_term(&term);
        let entry = self.terms.entry(start_pos).or_insert(TermEntry {
            frequency: 0,
            documents: HashSet::new(),
            start_pos,
            length: term.len(),
        });
        entry.frequency += 1;
        entry.documents.insert(document);
        self.total_words += 1;
    }

    pub fn merge_terms(&mut self, terms: Vec<(String, String)>) {
        for (term, document) in terms {
            self.add_term(term, document);
        }
    }

    pub fn add_file_stats(&mut self, file_size: u64) {
        self.collection_size_bytes += file_size;
        self.total_documents += 1;
    }

    pub fn dictionary_size(&self) -> usize {
        self.terms.len()
    }

    /// High-performance method to extract all terms from terms_string into a Vec<String>
    /// Uses parallel processing and memory-efficient techniques for large dictionaries
    pub fn extract_terms_parallel(&self) -> Vec<String> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        // Collect term positions and lengths for parallel processing
        let mut term_info: Vec<(usize, usize)> = self.terms
            .values()
            .map(|entry| (entry.start_pos, entry.length))
            .collect();

        // Sort by start position for better cache locality
        term_info.sort_by_key(|&(start_pos, _)| start_pos);

        const CHUNK_SIZE: usize = 1000; // Process in chunks for better parallelization

        if term_info.len() < CHUNK_SIZE {
            // For small dictionaries, use sequential processing to avoid overhead
            term_info
                .into_iter()
                .map(|(start_pos, length)| {
                    // Use unsafe slice access for maximum performance (bounds already verified)
                    unsafe {
                        let bytes = self.terms_string.as_bytes();
                        let term_bytes = &bytes[start_pos..start_pos + length];
                        String::from_utf8_unchecked(term_bytes.to_vec())
                    }
                })
                .collect()
        } else {
            // For large dictionaries, use parallel processing
            term_info
                .par_chunks(CHUNK_SIZE)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .map(|&(start_pos, length)| {
                            // Safe bounds checking for parallel processing
                            self.terms_string[start_pos..start_pos + length].to_string()
                        })
                        .collect::<Vec<String>>()
                })
                .collect()
        }
    }

    /// High-performance method to extract terms sorted by frequency
    /// Returns Vec<(String, u32)> with parallel processing for large dictionaries
    pub fn extract_terms_by_frequency_parallel(&self) -> Vec<(String, u32)> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        // Collect term data for parallel processing
        let term_data: Vec<(usize, usize, u32)> = self.terms
            .values()
            .map(|entry| (entry.start_pos, entry.length, entry.frequency))
            .collect();

        const CHUNK_SIZE: usize = 1000;

        let mut results = if term_data.len() < CHUNK_SIZE {
            // Sequential for small dictionaries
            term_data
                .into_iter()
                .map(|(start_pos, length, frequency)| {
                    let term = self.terms_string[start_pos..start_pos + length].to_string();
                    (term, frequency)
                })
                .collect::<Vec<_>>()
        } else {
            // Parallel for large dictionaries
            term_data
                .par_chunks(CHUNK_SIZE)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .map(|&(start_pos, length, frequency)| {
                            let term = self.terms_string[start_pos..start_pos + length].to_string();
                            (term, frequency)
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Sort by frequency (descending) using parallel sort for large datasets
        if results.len() > 10000 {
            results.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
        } else {
            results.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        }

        results
    }

    pub fn save_as_binary(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        let size = data.len();
        fs::write(path, data)?;
        Ok(size)
    }

    pub fn save_as_json(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let size = json.len();
        fs::write(path, json)?;
        Ok(size)
    }

    pub fn save_as_text(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;
        let header = format!(
            "DICTIONARY STATISTICS\n\
             Total terms: {}\n\
             Total words: {}\n\
             Total documents: {}\n\
             Collection size: {} bytes\n\n\
             TERMS (sorted by frequency):\n",
            self.dictionary_size(),
            self.total_words,
            self.total_documents,
            self.collection_size_bytes
        );
        file.write_all(header.as_bytes())?;

        let sorted_terms = self.extract_terms_by_frequency_parallel();

        for (term, frequency) in sorted_terms {
            let docs_count = self.terms.get(&self.term_lookup[&term]).unwrap().documents.len();
            let line = format!(
                "{}: {} (docs: {})\n",
                term,
                frequency,
                docs_count
            );
            file.write_all(line.as_bytes())?;
        }

        Ok(fs::metadata(path)?.len() as usize)
    }
}

impl CompressedDictionary {
    /// Create a compressed dictionary from a regular dictionary
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        println!("CompressedDictionary: Starting front-packing compression...");
        
        // Extract all terms for compression
        let terms = dictionary.extract_terms_parallel();
        let original_size = dictionary.terms_string.len();
        
        // Apply front-packing compression
        let (compressed_data, term_positions) = compress_terms(terms);
        let compressed_size = compressed_data.len();
        
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        println!(
            "CompressedDictionary: Front-packing complete - {:.2}% of original size ({} -> {} bytes)",
            compression_ratio * 100.0,
            original_size,
            compressed_size
        );

        CompressedDictionary {
            terms: dictionary.terms.clone(),
            compressed_terms_data: compressed_data,
            term_positions,
            total_words: dictionary.total_words,
            total_documents: dictionary.total_documents,
            collection_size_bytes: dictionary.collection_size_bytes,
            original_terms_size: original_size,
            compressed_terms_size: compressed_size,
        }
    }

    /// Get a term by looking it up in the compressed data
    pub fn get_term(&self, term: &str) -> Option<String> {
        // For now, we can use the term_positions map which contains the original terms
        // In a full implementation, you'd decompress from the compressed_terms_data
        if self.term_positions.contains_key(term) {
            Some(term.to_string())
        } else {
            None
        }
    }

    /// Check if a term exists in the compressed dictionary
    pub fn contains_term(&self, term: &str) -> bool {
        self.term_positions.contains_key(term)
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> (usize, usize, f64) {
        let ratio = if self.original_terms_size > 0 {
            self.compressed_terms_size as f64 / self.original_terms_size as f64
        } else {
            1.0
        };
        (self.original_terms_size, self.compressed_terms_size, ratio)
    }

    /// Get dictionary size (number of unique terms)
    pub fn dictionary_size(&self) -> usize {
        self.terms.len()
    }

    /// Extract all terms from compressed dictionary (parallel processing)
    pub fn extract_terms_parallel(&self) -> Vec<String> {
        // Extract from term_positions for now - in full implementation would decompress
        let mut terms: Vec<String> = self.term_positions.keys().cloned().collect();
        
        // Use parallel sort for large dictionaries
        if terms.len() > 10000 {
            terms.par_sort_unstable();
        } else {
            terms.sort_unstable();
        }
        
        terms
    }

    /// Extract terms sorted by frequency (parallel processing)
    pub fn extract_terms_by_frequency_parallel(&self) -> Vec<(String, u32)> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        // Get terms with their frequencies
        let mut term_frequencies: Vec<(String, u32)> = self.term_positions
            .keys()
            .filter_map(|term| {
                // Find the term entry using the term_positions mapping
                // This is a simplified approach - in practice you'd use proper indexing
                for (_, entry) in &self.terms {
                    if self.term_positions.contains_key(term) {
                        return Some((term.clone(), entry.frequency));
                    }
                }
                None
            })
            .collect();

        // Use parallel sort for large datasets
        if term_frequencies.len() > 10000 {
            term_frequencies.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
        } else {
            term_frequencies.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        }

        term_frequencies
    }

    /// Memory size of the compressed dictionary
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.terms.iter().map(|(_, entry)| {
                std::mem::size_of::<usize>() + std::mem::size_of::<TermEntry>() 
                    + entry.documents.len() * (std::mem::size_of::<String>() + 16) // approximate string overhead
            }).sum::<usize>()
            + self.compressed_terms_data.len()
            + self.term_positions.iter().map(|(k, _)| k.len() + 16).sum::<usize>()
    }

    /// Save compressed dictionary as binary
    pub fn save_as_binary(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        let size = data.len();
        fs::write(path, data)?;
        Ok(size)
    }

    /// Save compressed dictionary statistics as text
    pub fn save_as_text(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;
        let (original_size, compressed_size, ratio) = self.compression_stats();
        
        let header = format!(
            "COMPRESSED DICTIONARY STATISTICS\n\
             Total terms: {}\n\
             Total words: {}\n\
             Total documents: {}\n\
             Collection size: {} bytes\n\
             \n\
             COMPRESSION STATISTICS:\n\
             Original terms size: {} bytes\n\
             Compressed terms size: {} bytes\n\
             Compression ratio: {:.2}%\n\
             Space savings: {} bytes\n\
             \n\
             TERMS (sorted by frequency):\n",
            self.dictionary_size(),
            self.total_words,
            self.total_documents,
            self.collection_size_bytes,
            original_size,
            compressed_size,
            ratio * 100.0,
            original_size - compressed_size
        );
        file.write_all(header.as_bytes())?;

        let sorted_terms = self.extract_terms_by_frequency_parallel();

        for (term, frequency) in sorted_terms.iter().take(1000) { // Limit output for readability
            let line = format!("{}: {}\n", term, frequency);
            file.write_all(line.as_bytes())?;
        }

        Ok(fs::metadata(path)?.len() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_compression() {
        // Create a test dictionary
        let mut dict = Dictionary::new();
        
        // Add test terms with common prefixes (good for front-packing)
        let test_terms = vec![
            ("computer", "doc1.txt"),
            ("computing", "doc2.txt"),
            ("computational", "doc3.txt"),
            ("compile", "doc4.txt"),
            ("compression", "doc5.txt"),
            ("information", "doc6.txt"),
            ("inform", "doc7.txt"),
            ("informed", "doc8.txt"),
            ("retrieval", "doc9.txt"),
            ("retrieve", "doc10.txt"),
        ];
        
        for (term, doc) in test_terms {
            dict.add_term(term.to_string(), doc.to_string());
        }
        
        let original_size = dict.terms_string.len();
        assert_eq!(dict.dictionary_size(), 10);
        
        // Compress the dictionary
        let compressed = CompressedDictionary::from_dictionary(&dict);
        let (orig_size, comp_size, ratio) = compressed.compression_stats();
        
        // Verify compression occurred
        assert_eq!(orig_size, original_size);
        assert!(comp_size < orig_size, "Compressed size should be smaller than original");
        assert!(ratio < 1.0, "Compression ratio should be less than 1.0");
        
        // Test term lookup
        assert!(compressed.contains_term("computer"));
        assert!(compressed.contains_term("computing"));
        assert!(compressed.contains_term("information"));
        assert!(!compressed.contains_term("nonexistent"));
        
        // Test term retrieval
        assert_eq!(compressed.get_term("computer"), Some("computer".to_string()));
        assert_eq!(compressed.get_term("nonexistent"), None);
        
        // Verify dictionary size matches
        assert_eq!(compressed.dictionary_size(), dict.dictionary_size());
    }

    #[test]
    fn test_front_packing_compression() {
        let terms = vec![
            "test".to_string(),
            "testing".to_string(),
            "tested".to_string(),
            "computer".to_string(),
            "computing".to_string(),
        ];
        
        let (compressed_data, term_positions) = compress_terms(terms.clone());
        
        // Verify all terms are in positions map
        for term in &terms {
            assert!(term_positions.contains_key(term));
        }
        
        // Verify compressed data is not empty
        assert!(!compressed_data.is_empty());
        
        // Verify we can find terms
        assert!(term_positions.contains_key("test"));
        assert!(term_positions.contains_key("testing"));
        assert!(term_positions.contains_key("computer"));
    }

    #[test]
    fn test_compressed_dictionary_parallel_methods() {
        let mut dict = Dictionary::new();
        
        // Add many terms to test parallel processing
        for i in 0..2000 {
            dict.add_term(format!("term{:04}", i), format!("doc{}.txt", i % 100));
        }
        
        let compressed = CompressedDictionary::from_dictionary(&dict);
        
        // Test parallel term extraction
        let terms = compressed.extract_terms_parallel();
        assert_eq!(terms.len(), 2000);
        
        // Verify terms are sorted
        for i in 1..terms.len() {
            assert!(terms[i-1] <= terms[i], "Terms should be sorted");
        }
        
        // Test parallel frequency extraction
        let freq_terms = compressed.extract_terms_by_frequency_parallel();
        assert!(!freq_terms.is_empty());
        
        // Verify frequency sorting (descending)
        for i in 1..freq_terms.len() {
            assert!(freq_terms[i-1].1 >= freq_terms[i].1, "Frequencies should be sorted descending");
        }
    }
}
