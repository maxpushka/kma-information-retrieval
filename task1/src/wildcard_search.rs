use crate::{Dictionary, CompressedInvertedIndex, PermutationIndex, QueryParser, SuffixTree, TrigramIndex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Serialize, Deserialize)]
pub struct WildcardSearchEngine {
    inverted_index: CompressedInvertedIndex,
    suffix_tree: SuffixTree,
    permutation_index: PermutationIndex,
    trigram_index: TrigramIndex,
    dictionary: Dictionary,
}

impl WildcardSearchEngine {
    pub fn from_dictionary(dictionary: Dictionary) -> Self {
        println!("  WildcardSearchEngine: Starting construction");

        println!("  WildcardSearchEngine: Building inverted index...");
        let start = std::time::Instant::now();
        let inverted_index = CompressedInvertedIndex::from_dictionary(&dictionary);
        println!("    Inverted index built in {:.2?}", start.elapsed());

        println!("  WildcardSearchEngine: Building suffix tree...");
        let start = std::time::Instant::now();
        let suffix_tree = SuffixTree::from_dictionary(&dictionary);
        println!("    Suffix tree built in {:.2?}", start.elapsed());

        println!("  WildcardSearchEngine: Building permutation index...");
        let start = std::time::Instant::now();
        let permutation_index = PermutationIndex::from_dictionary(&dictionary);
        println!("    Permutation index built in {:.2?}", start.elapsed());

        println!("  WildcardSearchEngine: Building trigram index...");
        let start = std::time::Instant::now();
        let trigram_index = TrigramIndex::from_dictionary(&dictionary);
        println!("    Trigram index built in {:.2?}", start.elapsed());

        println!("  WildcardSearchEngine: Construction complete");
        WildcardSearchEngine {
            inverted_index,
            suffix_tree,
            permutation_index,
            trigram_index,
            dictionary,
        }
    }

    pub fn search(&self, query: &str) -> Result<HashSet<String>, String> {
        if query.is_empty() {
            return Err("Empty query".to_string());
        }

        if !query.contains('*') && !query.contains('?') {
            return self.exact_search(query);
        }

        self.wildcard_search(query)
    }

    fn exact_search(&self, term: &str) -> Result<HashSet<String>, String> {
        match self.inverted_index.search(term) {
            Ok(documents) => Ok(documents),
            Err(e) => Err(e),
        }
    }

    fn wildcard_search(&self, pattern: &str) -> Result<HashSet<String>, String> {
        let matching_terms = self.find_matching_terms(pattern)?;

        if matching_terms.is_empty() {
            return Ok(HashSet::new());
        }

        // Parallelize document lookup for large term sets
        let documents_sets: Vec<_> = if matching_terms.len() > 100 {
            // For large result sets, use parallel processing
            matching_terms
                .par_iter()
                .filter_map(|term| {
                    // Find the term in the dictionary by looking up its position
                    self.dictionary.terms.iter().find_map(|(start_pos, entry)| {
                        if let Some(dict_term) = self.dictionary.get_term(*start_pos) {
                            if dict_term == term {
                                Some(&entry.documents)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .map(|docs| docs.iter().cloned().collect::<HashSet<String>>())
                .collect()
        } else {
            // For small result sets, sequential is faster due to overhead
            matching_terms
                .iter()
                .filter_map(|term| {
                    // Find the term in the dictionary by looking up its position
                    self.dictionary.terms.iter().find_map(|(start_pos, entry)| {
                        if let Some(dict_term) = self.dictionary.get_term(*start_pos) {
                            if dict_term == term {
                                Some(&entry.documents)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .map(|docs| docs.iter().cloned().collect::<HashSet<String>>())
                .collect()
        };

        // Union all document sets
        let all_documents = documents_sets
            .into_iter()
            .fold(HashSet::new(), |mut acc, docs| {
                acc.extend(docs);
                acc
            });

        Ok(all_documents)
    }

    fn find_matching_terms(&self, pattern: &str) -> Result<HashSet<String>, String> {
        let wildcard_complexity = self.analyze_wildcard_complexity(pattern);

        match wildcard_complexity {
            WildcardComplexity::Simple => {
                if pattern.starts_with('*') && pattern.ends_with('*') {
                    Ok(self.suffix_tree.find_matching_terms(pattern))
                } else if pattern.starts_with('*') {
                    Ok(self.permutation_index.find_matching_terms(pattern))
                } else if pattern.ends_with('*') {
                    Ok(self.permutation_index.find_matching_terms(pattern))
                } else {
                    Ok(self.permutation_index.find_matching_terms(pattern))
                }
            }
            WildcardComplexity::Medium => {
                // Run multiple indices in parallel for medium complexity queries
                let (suffix_and_perm, trigram_results) = rayon::join(
                    || {
                        let (suffix_results, perm_results) = rayon::join(
                            || self.suffix_tree.find_matching_terms(pattern),
                            || self.permutation_index.find_matching_terms(pattern),
                        );
                        (suffix_results, perm_results)
                    },
                    || self.trigram_index.find_matching_terms(pattern),
                );
                let (suffix_results, perm_results) = suffix_and_perm;

                // Use intersection of suffix tree and permutation index (both are reliable)
                // Only include trigram results if they're not empty
                let reliable_intersection: HashSet<String> = suffix_results
                    .intersection(&perm_results)
                    .cloned()
                    .collect();

                if trigram_results.is_empty() {
                    // If trigram index has no results, use the reliable intersection
                    Ok(reliable_intersection)
                } else {
                    // Find intersection of all three for highest precision
                    let full_intersection: HashSet<String> = reliable_intersection
                        .intersection(&trigram_results)
                        .cloned()
                        .collect();
                    
                    if full_intersection.is_empty() {
                        // If full intersection is empty, use the reliable intersection
                        Ok(reliable_intersection)
                    } else {
                        Ok(full_intersection)
                    }
                }
            }
            WildcardComplexity::Complex => Ok(self.trigram_index.find_matching_terms(pattern)),
        }
    }

    fn analyze_wildcard_complexity(&self, pattern: &str) -> WildcardComplexity {
        let wildcard_count = pattern.chars().filter(|&c| c == '*' || c == '?').count();
        let total_chars = pattern.len();

        if wildcard_count == 0 {
            WildcardComplexity::Simple
        } else if wildcard_count == 1 {
            // Single wildcard patterns are simple regardless of position
            WildcardComplexity::Simple
        } else if wildcard_count <= 2 && (wildcard_count as f64 / total_chars as f64) < 0.5 {
            WildcardComplexity::Medium
        } else {
            WildcardComplexity::Complex
        }
    }

    pub fn memory_size(&self) -> WildcardMemoryStats {
        WildcardMemoryStats {
            inverted_index_size: self.inverted_index.memory_size(),
            suffix_tree_size: self.suffix_tree.memory_size(),
            permutation_index_size: self.permutation_index.memory_size(),
            trigram_index_size: self.trigram_index.memory_size(),
            total_size: self.inverted_index.memory_size()
                + self.suffix_tree.memory_size()
                + self.permutation_index.memory_size()
                + self.trigram_index.memory_size(),
        }
    }

    pub fn search_with_stats(&self, query: &str) -> WildcardSearchResult {
        let start_time = std::time::Instant::now();

        let result = self.search(query);
        let search_time = start_time.elapsed();

        let strategy = if query.contains('*') || query.contains('?') {
            match self.analyze_wildcard_complexity(query) {
                WildcardComplexity::Simple => "Permutation Index".to_string(),
                WildcardComplexity::Medium => "Hybrid (Multiple Indices)".to_string(),
                WildcardComplexity::Complex => "Trigram Index".to_string(),
            }
        } else {
            "Inverted Index".to_string()
        };

        match result {
            Ok(documents) => WildcardSearchResult {
                query: query.to_string(),
                documents,
                search_time,
                strategy,
                error: None,
            },
            Err(e) => WildcardSearchResult {
                query: query.to_string(),
                documents: HashSet::new(),
                search_time,
                strategy,
                error: Some(e),
            },
        }
    }
}

#[derive(Debug)]
enum WildcardComplexity {
    Simple,  // No wildcards or single prefix/suffix wildcard
    Medium,  // Few wildcards, mixed pattern
    Complex, // Many wildcards or complex pattern
}

#[derive(Debug)]
pub struct WildcardMemoryStats {
    pub inverted_index_size: usize,
    pub suffix_tree_size: usize,
    pub permutation_index_size: usize,
    pub trigram_index_size: usize,
    pub total_size: usize,
}

#[derive(Debug)]
pub struct WildcardSearchResult {
    pub query: String,
    pub documents: HashSet<String>,
    pub search_time: std::time::Duration,
    pub strategy: String,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Dictionary;

    fn create_test_dictionary() -> Dictionary {
        let mut dict = Dictionary::new();
        dict.add_term("hello".to_string(), "doc1.fb2".to_string());
        dict.add_term("help".to_string(), "doc2.fb2".to_string());
        dict.add_term("world".to_string(), "doc1.fb2".to_string());
        dict.add_term("wonderful".to_string(), "doc3.fb2".to_string());
        dict.add_term("testing".to_string(), "doc2.fb2".to_string());
        dict
    }

    #[test]
    fn test_exact_search() {
        let dict = create_test_dictionary();
        let engine = WildcardSearchEngine::from_dictionary(dict);

        let result = engine.search("hello").unwrap();
        assert!(result.contains("doc1.fb2"));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_prefix_wildcard() {
        let dict = create_test_dictionary();
        let engine = WildcardSearchEngine::from_dictionary(dict);

        let result = engine.search("hel*").unwrap();
        assert!(result.contains("doc1.fb2"));
        assert!(result.contains("doc2.fb2"));
    }

    #[test]
    fn test_suffix_wildcard() {
        let dict = create_test_dictionary();
        let engine = WildcardSearchEngine::from_dictionary(dict);

        let result = engine.search("*ing").unwrap();
        assert!(result.contains("doc2.fb2"));
    }

    #[test]
    fn test_middle_wildcard() {
        let dict = create_test_dictionary();
        let engine = WildcardSearchEngine::from_dictionary(dict);

        let result = engine.search("w*l").unwrap();
        assert!(result.contains("doc1.fb2"));
        assert!(result.contains("doc3.fb2"));
    }
}
