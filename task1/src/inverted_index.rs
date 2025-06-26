use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::dictionary::{Dictionary, CompressedDictionary};
use crate::query::{tokenize, QueryParser};

/// Variable-Byte encoding utilities for compressing document IDs
mod vb_encoding {
    /// Encode a single integer using Variable-Byte encoding
    pub fn encode_vb(mut n: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        while n >= 128 {
            bytes.push((n % 128) as u8);
            n /= 128;
        }
        bytes.push((n + 128) as u8); // Set continuation bit for last byte
        bytes
    }

    /// Decode Variable-Byte encoded bytes back to integers
    pub fn decode_vb(bytes: &[u8]) -> Vec<u32> {
        let mut numbers = Vec::new();
        let mut i = 0;
        
        while i < bytes.len() {
            let mut n = 0u32;
            let mut shift = 0;
            
            while i < bytes.len() {
                let byte = bytes[i];
                if byte < 128 {
                    n += (byte as u32) << shift;
                    shift += 7;
                    i += 1;
                } else {
                    n += ((byte - 128) as u32) << shift;
                    i += 1;
                    break;
                }
            }
            numbers.push(n);
        }
        numbers
    }

    /// Encode a list of integers with delta compression + VB encoding
    pub fn encode_delta_vb(mut numbers: Vec<u32>) -> Vec<u8> {
        if numbers.is_empty() {
            return Vec::new();
        }
        
        // Sort for better delta compression
        numbers.sort_unstable();
        
        let mut encoded = Vec::new();
        let mut prev = 0;
        
        for num in numbers {
            let delta = num - prev;
            encoded.extend(encode_vb(delta));
            prev = num;
        }
        encoded
    }

    /// Decode delta compressed + VB encoded bytes back to original numbers
    pub fn decode_delta_vb(bytes: &[u8]) -> Vec<u32> {
        let deltas = decode_vb(bytes);
        let mut numbers = Vec::new();
        let mut current = 0;
        
        for delta in deltas {
            current += delta;
            numbers.push(current);
        }
        numbers
    }
}

use vb_encoding::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct InvertedIndex {
    pub index: HashMap<String, HashSet<String>>,
    pub documents: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedInvertedIndex {
    /// Compressed posting lists: term -> compressed document IDs
    pub compressed_index: HashMap<String, Vec<u8>>,
    /// Document ID to document name mapping
    pub doc_id_to_name: Vec<String>,
    /// Document name to ID mapping for fast lookups
    pub doc_name_to_id: HashMap<String, u32>,
    /// Total memory used by compressed data
    pub compressed_size: usize,
    /// Original uncompressed size for comparison
    pub uncompressed_size: usize,
}

impl InvertedIndex {
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        // Use parallel processing for large dictionaries
        const PARALLEL_THRESHOLD: usize = 1000;
        
        if dictionary.terms.len() < PARALLEL_THRESHOLD {
            // Sequential processing for small dictionaries
            let mut index = HashMap::new();
            let mut documents = HashSet::new();

            for (&start_pos, term_entry) in &dictionary.terms {
                let term = dictionary.get_term(start_pos).unwrap().to_string();
                index.insert(term, term_entry.documents.clone());
                for doc in &term_entry.documents {
                    documents.insert(doc.clone());
                }
            }

            let mut documents: Vec<String> = documents.into_iter().collect();
            documents.sort();
            InvertedIndex { index, documents }
        } else {
            // Parallel processing for large dictionaries
            let term_entries: Vec<_> = dictionary.terms.iter().collect();
            
            // Build index in parallel
            let index: HashMap<String, HashSet<String>> = term_entries
                .par_iter()
                .map(|(&start_pos, term_entry)| {
                    let term = dictionary.get_term(start_pos).unwrap().to_string();
                    (term, term_entry.documents.clone())
                })
                .collect();

            // Collect all unique documents in parallel
            let all_docs: HashSet<String> = term_entries
                .par_iter()
                .flat_map(|(_, term_entry)| term_entry.documents.par_iter().cloned())
                .collect();

            let mut documents: Vec<String> = all_docs.into_iter().collect();
            
            // Use parallel sort for large document collections
            if documents.len() > 10000 {
                documents.par_sort_unstable();
            } else {
                documents.sort_unstable();
            }

            InvertedIndex { index, documents }
        }
    }

    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .index
                .iter()
                .map(|(k, v)| {
                    k.len()
                        + std::mem::size_of::<HashSet<String>>()
                        + v.iter().map(|s| s.len()).sum::<usize>()
                })
                .sum::<usize>()
            + self.documents.iter().map(|d| d.len()).sum::<usize>()
    }

    fn parse_or_expr(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        let mut result = self.parse_and_expr(tokens, pos)?;

        while *pos < tokens.len() && tokens[*pos] == "or" {
            *pos += 1;
            let right = self.parse_and_expr(tokens, pos)?;
            result = result.union(&right).cloned().collect();
        }

        Ok(result)
    }

    fn parse_and_expr(
        &self,
        tokens: &[String],
        pos: &mut usize,
    ) -> Result<HashSet<String>, String> {
        let mut result = self.parse_not_expr(tokens, pos)?;

        while *pos < tokens.len() && tokens[*pos] == "and" {
            *pos += 1;
            let right = self.parse_not_expr(tokens, pos)?;
            result = result.intersection(&right).cloned().collect();
        }

        Ok(result)
    }

    fn parse_not_expr(
        &self,
        tokens: &[String],
        pos: &mut usize,
    ) -> Result<HashSet<String>, String> {
        if *pos < tokens.len() && tokens[*pos] == "not" {
            *pos += 1;
            let result = self.parse_primary(tokens, pos)?;
            let all_docs: HashSet<String> = self.documents.iter().cloned().collect();
            Ok(all_docs.difference(&result).cloned().collect())
        } else {
            self.parse_primary(tokens, pos)
        }
    }

    fn parse_primary(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of query".to_string());
        }

        if tokens[*pos] == "(" {
            *pos += 1;
            let result = self.parse_or_expr(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("Missing closing parenthesis".to_string());
            }
            *pos += 1;
            Ok(result)
        } else {
            let term = &tokens[*pos];
            *pos += 1;
            self.search_term(term)
        }
    }
}

impl QueryParser for InvertedIndex {
    type Result = HashSet<String>;
    type Error = String;

    fn search(&self, query: &str) -> Result<Self::Result, Self::Error> {
        let tokens = tokenize(&query.to_lowercase())?;
        let mut pos = 0;
        self.parse_or_expr(&tokens, &mut pos)
    }

    fn search_term(&self, term: &str) -> Result<Self::Result, Self::Error> {
        if let Some(docs) = self.index.get(term) {
            Ok(docs.clone())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}

impl CompressedInvertedIndex {
    /// Create a compressed inverted index from a regular inverted index
    pub fn from_inverted_index(index: &InvertedIndex) -> Self {
        println!("CompressedInvertedIndex: Starting compression...");
        
        // Create document ID mappings
        let mut doc_id_to_name = index.documents.clone();
        doc_id_to_name.sort(); // Ensure consistent ordering
        
        let doc_name_to_id: HashMap<String, u32> = doc_id_to_name
            .iter()
            .enumerate()
            .map(|(id, name)| (name.clone(), id as u32))
            .collect();

        let mut compressed_index = HashMap::new();
        let mut total_compressed_size = 0;
        let mut total_uncompressed_size = 0;

        // Use parallel processing for large indexes
        const PARALLEL_THRESHOLD: usize = 1000;
        
        if index.index.len() < PARALLEL_THRESHOLD {
            // Sequential compression for small indexes
            for (term, docs) in &index.index {
                let doc_ids: Vec<u32> = docs
                    .iter()
                    .filter_map(|doc| doc_name_to_id.get(doc).copied())
                    .collect();
                
                let uncompressed_size = doc_ids.len() * 4; // 4 bytes per u32
                let compressed_bytes = encode_delta_vb(doc_ids);
                
                total_uncompressed_size += uncompressed_size;
                total_compressed_size += compressed_bytes.len();
                
                compressed_index.insert(term.clone(), compressed_bytes);
            }
        } else {
            // Parallel compression for large indexes
            let entries: Vec<_> = index.index.iter().collect();
            
            let compressed_entries: HashMap<String, Vec<u8>> = entries
                .par_iter()
                .map(|(term, docs)| {
                    let doc_ids: Vec<u32> = docs
                        .iter()
                        .filter_map(|doc| doc_name_to_id.get(doc).copied())
                        .collect();
                    
                    let compressed_bytes = encode_delta_vb(doc_ids);
                    ((*term).clone(), compressed_bytes)
                })
                .collect();

            // Calculate sizes
            for (term, docs) in &index.index {
                let uncompressed_size = docs.len() * 4; // 4 bytes per u32
                let compressed_size = compressed_entries.get(term).map_or(0, |v| v.len());
                
                total_uncompressed_size += uncompressed_size;
                total_compressed_size += compressed_size;
            }
            
            compressed_index = compressed_entries;
        }

        let compression_ratio = if total_uncompressed_size > 0 {
            total_compressed_size as f64 / total_uncompressed_size as f64
        } else {
            1.0
        };

        println!(
            "CompressedInvertedIndex: Compression complete - {:.2}% of original size ({} -> {} bytes)",
            compression_ratio * 100.0,
            total_uncompressed_size,
            total_compressed_size
        );

        CompressedInvertedIndex {
            compressed_index,
            doc_id_to_name,
            doc_name_to_id,
            compressed_size: total_compressed_size,
            uncompressed_size: total_uncompressed_size,
        }
    }

    /// Create compressed inverted index directly from dictionary (most efficient)
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        println!("CompressedInvertedIndex: Creating compressed index from dictionary...");
        
        // Build regular index first, then compress
        let regular_index = InvertedIndex::from_dictionary(dictionary);
        Self::from_inverted_index(&regular_index)
    }
    
    /// Create compressed inverted index from compressed dictionary
    pub fn from_compressed_dictionary(dictionary: &CompressedDictionary) -> Self {
        println!("CompressedInvertedIndex: Creating compressed index from compressed dictionary...");
        
        // Use parallel processing for large dictionaries
        const PARALLEL_THRESHOLD: usize = 1000;
        
        if dictionary.terms.len() < PARALLEL_THRESHOLD {
            // Sequential processing for small dictionaries
            let mut index = HashMap::new();
            let mut documents = HashSet::new();

            for (&start_pos, term_entry) in &dictionary.terms {
                if let Some(term) = dictionary.get_term(start_pos) {
                    index.insert(term.to_string(), term_entry.documents.clone());
                    for doc in &term_entry.documents {
                        documents.insert(doc.clone());
                    }
                }
            }

            let mut documents: Vec<String> = documents.into_iter().collect();
            documents.sort();
            
            let regular_index = InvertedIndex { index, documents };
            Self::from_inverted_index(&regular_index)
        } else {
            // Parallel processing for large dictionaries
            let term_entries: Vec<_> = dictionary.terms.iter().collect();
            
            // Build index in parallel
            let index: HashMap<String, HashSet<String>> = term_entries
                .par_iter()
                .filter_map(|(&start_pos, term_entry)| {
                    if let Some(term) = dictionary.get_term(start_pos) {
                        Some((term.to_string(), term_entry.documents.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            // Collect all unique documents in parallel
            let all_docs: HashSet<String> = term_entries
                .par_iter()
                .flat_map(|(_, term_entry)| term_entry.documents.par_iter().cloned())
                .collect();

            let mut documents: Vec<String> = all_docs.into_iter().collect();
            
            // Use parallel sort for large document collections
            if documents.len() > 10000 {
                documents.par_sort_unstable();
            } else {
                documents.sort_unstable();
            }

            let regular_index = InvertedIndex { index, documents };
            Self::from_inverted_index(&regular_index)
        }
    }

    /// Decompress posting list for a specific term
    pub fn get_documents_for_term(&self, term: &str) -> Option<Vec<String>> {
        if let Some(compressed_bytes) = self.compressed_index.get(term) {
            let doc_ids = decode_delta_vb(compressed_bytes);
            let documents: Vec<String> = doc_ids
                .into_iter()
                .filter_map(|id| self.doc_id_to_name.get(id as usize).cloned())
                .collect();
            Some(documents)
        } else {
            None
        }
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> (usize, usize, f64) {
        let ratio = if self.uncompressed_size > 0 {
            self.compressed_size as f64 / self.uncompressed_size as f64
        } else {
            1.0
        };
        (self.uncompressed_size, self.compressed_size, ratio)
    }

    /// Memory size of the compressed index
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.compressed_index
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + self.doc_id_to_name.iter().map(|s| s.len()).sum::<usize>()
            + self.doc_name_to_id
                .iter()
                .map(|(k, _)| k.len() + 4)
                .sum::<usize>()
    }
}

impl QueryParser for CompressedInvertedIndex {
    type Result = HashSet<String>;
    type Error = String;

    fn search(&self, query: &str) -> Result<Self::Result, Self::Error> {
        let tokens = tokenize(&query.to_lowercase())?;
        let mut pos = 0;
        self.parse_or_expr(&tokens, &mut pos)
    }

    fn search_term(&self, term: &str) -> Result<Self::Result, Self::Error> {
        if let Some(docs) = self.get_documents_for_term(term) {
            Ok(docs.into_iter().collect())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}

// Implement the same query parsing methods for CompressedInvertedIndex
impl CompressedInvertedIndex {
    fn parse_or_expr(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        let mut result = self.parse_and_expr(tokens, pos)?;

        while *pos < tokens.len() && tokens[*pos] == "or" {
            *pos += 1;
            let right = self.parse_and_expr(tokens, pos)?;
            result = result.union(&right).cloned().collect();
        }

        Ok(result)
    }

    fn parse_and_expr(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        let mut result = self.parse_not_expr(tokens, pos)?;

        while *pos < tokens.len() && tokens[*pos] == "and" {
            *pos += 1;
            let right = self.parse_not_expr(tokens, pos)?;
            result = result.intersection(&right).cloned().collect();
        }

        Ok(result)
    }

    fn parse_not_expr(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        if *pos < tokens.len() && tokens[*pos] == "not" {
            *pos += 1;
            let result = self.parse_primary(tokens, pos)?;
            let all_docs: HashSet<String> = self.doc_id_to_name.iter().cloned().collect();
            Ok(all_docs.difference(&result).cloned().collect())
        } else {
            self.parse_primary(tokens, pos)
        }
    }

    fn parse_primary(&self, tokens: &[String], pos: &mut usize) -> Result<HashSet<String>, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of query".to_string());
        }

        if tokens[*pos] == "(" {
            *pos += 1;
            let result = self.parse_or_expr(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("Missing closing parenthesis".to_string());
            }
            *pos += 1;
            Ok(result)
        } else {
            let term = &tokens[*pos];
            *pos += 1;
            self.search_term(term)
        }
    }
}
