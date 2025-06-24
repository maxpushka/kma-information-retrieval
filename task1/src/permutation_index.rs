use crate::Dictionary;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
pub struct PermutationIndex {
    index: HashMap<String, HashSet<String>>,
}

impl PermutationIndex {
    pub fn new() -> Self {
        PermutationIndex {
            index: HashMap::new(),
        }
    }

    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        println!(
            "      PermutationIndex: Processing {} terms in parallel",
            dictionary.terms.len()
        );

        let index = Arc::new(Mutex::new(HashMap::new()));
        let terms = dictionary.extract_terms_parallel();

        // Process terms in parallel chunks
        let chunk_size = 1000;
        let chunks: Vec<_> = terms.chunks(chunk_size).collect();

        chunks
            .par_iter()
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rotations = HashMap::new();

                // Generate rotations for this chunk locally (no mutex contention)
                for term in *chunk {
                    let rotations = Self::generate_rotations_static(term);
                    for rotation in rotations {
                        local_rotations
                            .entry(rotation)
                            .or_insert_with(HashSet::new)
                            .insert(term.clone());
                    }
                }

                // Merge local results into global index (minimal mutex time)
                if let Ok(mut global_index) = index.lock() {
                    for (rotation, terms_set) in local_rotations {
                        global_index
                            .entry(rotation)
                            .or_insert_with(HashSet::new)
                            .extend(terms_set);
                    }
                }

                let processed = (chunk_idx + 1) * chunk_size;
                if processed % 5000 == 0 || chunk_idx == chunks.len() - 1 {
                    println!(
                        "      PermutationIndex: Processed ~{} terms",
                        processed.min(terms.len())
                    );
                }
            });

        let final_index = Arc::try_unwrap(index).unwrap().into_inner().unwrap();
        println!(
            "      PermutationIndex: Complete - {} terms, {} rotations",
            terms.len(),
            final_index.len()
        );

        PermutationIndex { index: final_index }
    }

    /// Generate rotations for a single term
    pub(crate) fn generate_rotations_static(term: &str) -> Vec<String> {
        let mut rotations = Vec::new();
        let term_with_marker = format!("{}$", term);
        let chars: Vec<char> = term_with_marker.chars().collect();

        for i in 0..chars.len() {
            let rotation: String = chars[i..].iter().chain(chars[..i].iter()).collect();
            rotations.push(rotation);
        }

        rotations
    }

    pub fn find_matching_terms(&self, pattern: &str) -> HashSet<String> {
        if pattern.is_empty() {
            return HashSet::new();
        }

        let mut results = HashSet::new();

        if pattern.contains('*') {
            let pattern_with_marker = if pattern.ends_with('*') {
                pattern.replacen('*', "$", 1)
            } else if pattern.starts_with('*') {
                format!("${}", &pattern[1..])
            } else {
                let parts: Vec<&str> = pattern.split('*').collect();
                if parts.len() == 2 {
                    format!("{}${}", parts[1], parts[0])
                } else {
                    pattern.to_string()
                }
            };

            for (rotation, terms) in &self.index {
                if self.matches_wildcard_pattern(rotation, &pattern_with_marker) {
                    results.extend(terms.iter().cloned());
                }
            }
        } else {
            let pattern_with_marker = format!("{}$", pattern);
            for (rotation, terms) in &self.index {
                if rotation.starts_with(&pattern_with_marker) {
                    results.extend(terms.iter().cloned());
                }
            }
        }

        results
    }

    fn matches_wildcard_pattern(&self, rotation: &str, pattern: &str) -> bool {
        if pattern.contains('$') {
            if pattern.ends_with('$') {
                let prefix = &pattern[..pattern.len() - 1];
                rotation.starts_with(prefix)
            } else if pattern.starts_with('$') {
                let suffix = &pattern[1..];
                rotation.ends_with(suffix)
            } else {
                let parts: Vec<&str> = pattern.split('$').collect();
                if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                    rotation.starts_with(parts[1]) && rotation.contains(parts[0])
                } else {
                    false
                }
            }
        } else {
            rotation.contains(pattern)
        }
    }

    pub fn memory_size(&self) -> usize {
        let mut size = std::mem::size_of::<PermutationIndex>();

        for (key, values) in &self.index {
            size += std::mem::size_of::<String>() + key.len();
            size += std::mem::size_of::<HashSet<String>>();
            for value in values {
                size += std::mem::size_of::<String>() + value.len();
            }
        }

        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Dictionary;

    #[test]
    fn test_permutation_index_prefix() {
        let mut dict = Dictionary::new();
        dict.add_term("hello".to_string(), "doc1".to_string());
        dict.add_term("help".to_string(), "doc1".to_string());
        dict.add_term("world".to_string(), "doc1".to_string());

        let perm_index = PermutationIndex::from_dictionary(&dict);

        let results = perm_index.find_matching_terms("hel*");
        assert!(results.contains("hello"));
        assert!(results.contains("help"));
        assert!(!results.contains("world"));
    }

    #[test]
    fn test_permutation_index_suffix() {
        let mut dict = Dictionary::new();
        dict.add_term("testing".to_string(), "doc1".to_string());
        dict.add_term("running".to_string(), "doc1".to_string());
        dict.add_term("hello".to_string(), "doc1".to_string());

        let perm_index = PermutationIndex::from_dictionary(&dict);

        let results = perm_index.find_matching_terms("*ing");
        assert!(results.contains("testing"));
        assert!(results.contains("running"));
        assert!(!results.contains("hello"));
    }

    #[test]
    fn test_permutation_index_middle() {
        let mut dict = Dictionary::new();
        dict.add_term("hello".to_string(), "doc1".to_string());
        dict.add_term("world".to_string(), "doc1".to_string());
        dict.add_term("wonderful".to_string(), "doc1".to_string());

        let perm_index = PermutationIndex::from_dictionary(&dict);

        let results = perm_index.find_matching_terms("w*l");
        assert!(results.contains("world"));
        assert!(results.contains("wonderful"));
        assert!(!results.contains("hello"));
    }
}
