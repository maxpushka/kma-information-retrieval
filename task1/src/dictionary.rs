use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;

/// Front-packing compression for concatenated string dictionary
mod front_packing {
    /// Compress terms into a concatenated string with front-packing
    /// Returns (concatenated_string, term_offsets_array)
    pub fn compress_terms_to_string(mut terms: Vec<String>) -> (String, Vec<(usize, usize, usize, usize)>) {
        if terms.is_empty() {
            return (String::new(), Vec::new());
        }

        // Sort terms lexicographically for front-packing
        terms.sort();

        let mut concatenated_string = String::new();
        let mut term_offsets = Vec::with_capacity(terms.len());
        let mut i = 0;

        while i < terms.len() {
            // Find optimal block for front-packing
            let (block_end, prefix_len) = find_optimal_block(&terms, i);
            let block_terms = &terms[i..block_end];

            if prefix_len > 0 && block_terms.len() > 1 {
                // Front-pack this block: store prefix once, then suffixes
                let prefix = &block_terms[0][..prefix_len];

                // Store prefix once
                let prefix_start = concatenated_string.len();
                concatenated_string.push_str(prefix);

                // Store each term's suffix and track reconstruction info
                for term in block_terms {
                    let suffix = &term[prefix_len..];
                    let suffix_start = concatenated_string.len();
                    concatenated_string.push_str(suffix);
                    let suffix_end = concatenated_string.len();

                    // Store reconstruction info: (prefix_start, prefix_len, suffix_start, suffix_len)
                    term_offsets.push((prefix_start, prefix_len, suffix_start, suffix_end - suffix_start));
                }
            } else {
                // No compression benefit, store terms normally
                for term in block_terms {
                    let start_pos = concatenated_string.len();
                    concatenated_string.push_str(term);
                    let end_pos = concatenated_string.len();
                    // For non-compressed terms: (term_start, term_len, 0, 0) - no separate prefix/suffix
                    term_offsets.push((start_pos, end_pos - start_pos, 0, 0));
                }
            }

            i = block_end;
        }

        (concatenated_string, term_offsets)
    }

    /// Find the optimal block size and prefix length for front-packing
    fn find_optimal_block(terms: &[String], start: usize) -> (usize, usize) {
        if start >= terms.len() {
            return (start, 0);
        }

        let mut best_end = start + 1;
        let mut best_prefix_len = 0;
        let mut best_savings = 0i32;

        // Try different block sizes and prefix lengths
        for end in (start + 2)..=std::cmp::min(start + 16, terms.len()) {
            if let Some(prefix_len) = find_common_prefix(&terms[start..end]) {
                if prefix_len == 0 {
                    continue;
                }

                // Calculate space savings: original size vs compressed size
                let original_size: usize = terms[start..end].iter().map(|s| s.len()).sum();
                // Compressed size = prefix_len + sum of suffix lengths
                let compressed_size = prefix_len + terms[start..end].iter().map(|s| s.len() - prefix_len).sum::<usize>();

                let savings = original_size as i32 - compressed_size as i32;

                if savings > best_savings {
                    best_savings = savings;
                    best_end = end;
                    best_prefix_len = prefix_len;
                }
            }
        }

        (best_end, best_prefix_len)
    }

    /// Find the longest common prefix among terms
    fn find_common_prefix(terms: &[String]) -> Option<usize> {
        if terms.len() < 2 {
            return Some(0);
        }

        let first = &terms[0];
        let mut prefix_len = 0;

        for (i, ch) in first.char_indices() {
            if terms.iter().all(|term| {
                term.chars().nth(i).map_or(false, |c| c == ch)
            }) {
                prefix_len = i + ch.len_utf8();
            } else {
                break;
            }
        }

        Some(prefix_len)
    }

}

use front_packing::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermEntry {
    pub frequency: u32,
    pub documents: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dictionary {
    pub terms: HashMap<String, TermEntry>,
    pub total_words: u64,
    pub total_documents: u32,
    pub collection_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDictionary {
    /// Concatenated string containing all terms with front-packing
    pub terms_string: String,
    /// Term reconstruction info: (prefix_start, prefix_len, suffix_start, suffix_len)
    pub term_offsets: Vec<(usize, usize, usize, usize)>,
    /// Sorted list of terms for binary search
    pub sorted_terms: Vec<String>,
    /// Term entries mapped by index (parallel to sorted_terms)
    pub term_entries: Vec<TermEntry>,
    /// Statistics
    pub total_words: u64,
    pub total_documents: u32,
    pub collection_size_bytes: u64,
    /// Compression statistics
    pub original_terms_size: usize,
    pub compressed_terms_size: usize,
}

impl Dictionary {
    pub fn new() -> Self {
        Dictionary {
            terms: HashMap::new(),
            total_words: 0,
            total_documents: 0,
            collection_size_bytes: 0,
        }
    }

    pub fn add_term(&mut self, term: String, document: String) {
        let entry = self.terms.entry(term).or_insert(TermEntry {
            frequency: 0,
            documents: HashSet::new(),
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

    /// High-performance method to extract all terms into a Vec<String>
    /// Uses parallel processing for large dictionaries
    pub fn extract_terms_parallel(&self) -> Vec<String> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        let mut terms: Vec<String> = self.terms.keys().cloned().collect();

        const CHUNK_SIZE: usize = 1000;

        if terms.len() > CHUNK_SIZE {
            terms.par_sort_unstable();
        } else {
            terms.sort_unstable();
        }

        terms
    }

    /// High-performance method to extract terms sorted by frequency
    /// Returns Vec<(String, u32)> with parallel processing for large dictionaries
    pub fn extract_terms_by_frequency_parallel(&self) -> Vec<(String, u32)> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<(String, u32)> = self.terms
            .iter()
            .map(|(term, entry)| (term.clone(), entry.frequency))
            .collect();

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
            let docs_count = self.terms.get(&term).unwrap().documents.len();
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

        // Extract all terms and sort them
        let terms = dictionary.extract_terms_parallel();
        let original_size = terms.iter().map(|term| term.len()).sum::<usize>();

        // Apply front-packing compression to create concatenated string
        let (terms_string, term_offsets) = compress_terms_to_string(terms.clone());
        let compressed_size = terms_string.len();

        // Create parallel arrays for binary search
        let mut sorted_terms = terms.clone();
        sorted_terms.sort();

        let mut term_entries = Vec::with_capacity(sorted_terms.len());
        for term in &sorted_terms {
            if let Some(entry) = dictionary.terms.get(term) {
                term_entries.push(entry.clone());
            }
        }

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
            terms_string,
            term_offsets,
            sorted_terms,
            term_entries,
            total_words: dictionary.total_words,
            total_documents: dictionary.total_documents,
            collection_size_bytes: dictionary.collection_size_bytes,
            original_terms_size: original_size,
            compressed_terms_size: compressed_size,
        }
    }

    /// Get a term by binary search in the sorted terms
    pub fn get_term(&self, term: &str) -> Option<String> {
        match self.sorted_terms.binary_search(&term.to_string()) {
            Ok(index) => {
                let (prefix_start, prefix_len, suffix_start, suffix_len) = self.term_offsets[index];

                if suffix_start == 0 && suffix_len == 0 {
                    // Non-compressed term: prefix_start is actually term start, prefix_len is term length
                    Some(self.terms_string[prefix_start..prefix_start + prefix_len].to_string())
                } else {
                    // Front-packed term: reconstruct from prefix + suffix
                    let prefix = &self.terms_string[prefix_start..prefix_start + prefix_len];
                    let suffix = &self.terms_string[suffix_start..suffix_start + suffix_len];
                    Some(format!("{}{}", prefix, suffix))
                }
            }
            Err(_) => None,
        }
    }

    /// Check if a term exists using binary search
    pub fn contains_term(&self, term: &str) -> bool {
        self.sorted_terms.binary_search(&term.to_string()).is_ok()
    }

    /// Get term entry by binary search
    pub fn get_term_entry(&self, term: &str) -> Option<&TermEntry> {
        match self.sorted_terms.binary_search(&term.to_string()) {
            Ok(index) => self.term_entries.get(index),
            Err(_) => None,
        }
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
        self.sorted_terms.len()
    }

    /// Extract all terms from compressed dictionary (already sorted)
    pub fn extract_terms_parallel(&self) -> Vec<String> {
        self.sorted_terms.clone()
    }

    /// Extract terms sorted by frequency (parallel processing)
    pub fn extract_terms_by_frequency_parallel(&self) -> Vec<(String, u32)> {
        if self.sorted_terms.is_empty() {
            return Vec::new();
        }

        // Create term-frequency pairs using parallel arrays
        let mut term_frequencies: Vec<(String, u32)> = self.sorted_terms
            .iter()
            .zip(self.term_entries.iter())
            .map(|(term, entry)| (term.clone(), entry.frequency))
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
            + self.terms_string.len()
            + self.term_offsets.len() * std::mem::size_of::<(usize, usize, usize, usize)>()
            + self.sorted_terms.iter().map(|s| s.len()).sum::<usize>()
            + self.term_entries.iter().map(|entry| {
                std::mem::size_of::<TermEntry>()
                    + entry.documents.len() * (std::mem::size_of::<String>() + 16)
            }).sum::<usize>()
    }

    /// Save compressed dictionary as JSON
    pub fn save_as_json(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let size = json.len();
        fs::write(path, json)?;
        Ok(size)
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

        let original_size = dict.terms.keys().map(|k| k.len()).sum::<usize>();
        assert_eq!(dict.dictionary_size(), 10);

        // Compress the dictionary
        let compressed = CompressedDictionary::from_dictionary(&dict);
        let (orig_size, comp_size, ratio) = compressed.compression_stats();

        // Verify compression occurred
        assert_eq!(orig_size, original_size);
        assert!(comp_size < orig_size, "Compressed size should be smaller than original");
        assert!(ratio < 1.0, "Compression ratio should be less than 1.0");

        // Test term lookup with binary search
        assert!(compressed.contains_term("computer"));
        assert!(compressed.contains_term("computing"));
        assert!(compressed.contains_term("information"));
        assert!(!compressed.contains_term("nonexistent"));

        // Test term retrieval
        assert_eq!(compressed.get_term("computer"), Some("computer".to_string()));
        assert_eq!(compressed.get_term("nonexistent"), None);

        // Test term entry retrieval
        assert!(compressed.get_term_entry("computer").is_some());
        assert!(compressed.get_term_entry("nonexistent").is_none());

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

        let (concatenated_string, term_offsets) = compress_terms_to_string(terms.clone());

        // Verify we have offsets for all terms
        assert_eq!(term_offsets.len(), terms.len());

        // Verify concatenated string is not empty
        assert!(!concatenated_string.is_empty());

        // Verify we can extract terms using offsets
        for (_, &(prefix_start, prefix_len, suffix_start, suffix_len)) in term_offsets.iter().enumerate() {
            let extracted_term = if suffix_start == 0 && suffix_len == 0 {
                // Non-compressed term
                concatenated_string[prefix_start..prefix_start + prefix_len].to_string()
            } else {
                // Front-packed term
                let prefix = &concatenated_string[prefix_start..prefix_start + prefix_len];
                let suffix = &concatenated_string[suffix_start..suffix_start + suffix_len];
                format!("{}{}", prefix, suffix)
            };
            assert!(terms.contains(&extracted_term));
        }
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
