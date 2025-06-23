use crate::Dictionary;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrigramIndex {
    index: HashMap<String, HashSet<String>>,
}

impl TrigramIndex {
    pub fn new() -> Self {
        TrigramIndex {
            index: HashMap::new(),
        }
    }

    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        println!(
            "      TrigramIndex: Processing {} terms in parallel",
            dictionary.terms.len()
        );

        let index = Arc::new(Mutex::new(HashMap::new()));
        let terms: Vec<&String> = dictionary.terms.keys().collect();

        // Process terms in parallel chunks
        let chunk_size = 1000;
        let chunks: Vec<_> = terms.chunks(chunk_size).collect();

        chunks
            .par_iter()
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_trigrams = HashMap::new();

                // Generate trigrams for this chunk locally (no mutex contention)
                for term in *chunk {
                    let trigrams = Self::generate_trigrams_static(term);
                    for trigram in trigrams {
                        local_trigrams
                            .entry(trigram)
                            .or_insert_with(HashSet::new)
                            .insert(term.to_string());
                    }
                }

                // Merge local results into global index (minimal mutex time)
                if let Ok(mut global_index) = index.lock() {
                    for (trigram, terms_set) in local_trigrams {
                        global_index
                            .entry(trigram)
                            .or_insert_with(HashSet::new)
                            .extend(terms_set);
                    }
                }

                let processed = (chunk_idx + 1) * chunk_size;
                if processed % 5000 == 0 || chunk_idx == chunks.len() - 1 {
                    println!(
                        "      TrigramIndex: Processed ~{} terms",
                        processed.min(terms.len())
                    );
                }
            });

        let final_index = Arc::try_unwrap(index).unwrap().into_inner().unwrap();
        println!(
            "      TrigramIndex: Complete - {} terms, {} trigrams",
            terms.len(),
            final_index.len()
        );

        TrigramIndex { index: final_index }
    }

    pub(crate) fn generate_trigrams_static(term: &str) -> Vec<String> {
        if term.len() < 3 {
            return vec![format!("$${}$$", term)];
        }

        let mut trigrams = Vec::new();
        let padded_term = format!("$${}", term);
        let chars: Vec<char> = padded_term.chars().collect();

        for i in 0..=chars.len().saturating_sub(3) {
            let trigram: String = chars[i..i + 3].iter().collect();
            trigrams.push(trigram);
        }

        trigrams
    }

    pub fn find_matching_terms(&self, pattern: &str) -> HashSet<String> {
        if pattern.is_empty() {
            return HashSet::new();
        }

        if !pattern.contains('*') && !pattern.contains('?') {
            return self.find_exact_match(pattern);
        }

        let required_trigrams = self.extract_required_trigrams(pattern);

        if required_trigrams.is_empty() {
            return HashSet::new();
        }

        let mut candidates: Option<HashSet<String>> = None;

        for trigram in required_trigrams {
            if let Some(terms) = self.index.get(&trigram) {
                match candidates {
                    None => candidates = Some(terms.clone()),
                    Some(ref mut existing) => {
                        existing.retain(|term| terms.contains(term));
                    }
                }
            } else {
                return HashSet::new();
            }
        }

        let candidates = candidates.unwrap_or_default();

        candidates
            .into_iter()
            .filter(|term| self.matches_pattern(term, pattern))
            .collect()
    }

    fn find_exact_match(&self, pattern: &str) -> HashSet<String> {
        let pattern_trigrams = Self::generate_trigrams_static(pattern);
        let mut candidates: Option<HashSet<String>> = None;

        for trigram in pattern_trigrams {
            if let Some(terms) = self.index.get(&trigram) {
                match candidates {
                    None => candidates = Some(terms.clone()),
                    Some(ref mut existing) => {
                        existing.retain(|term| terms.contains(term));
                    }
                }
            } else {
                return HashSet::new();
            }
        }

        candidates
            .unwrap_or_default()
            .into_iter()
            .filter(|term| *term == pattern)
            .collect()
    }

    fn extract_required_trigrams(&self, pattern: &str) -> Vec<String> {
        let mut trigrams = Vec::new();
        let chars: Vec<char> = pattern.chars().collect();

        let mut i = 0;
        while i + 2 < chars.len() {
            let window = &chars[i..i + 3];

            if !window.iter().any(|&c| c == '*' || c == '?') {
                let trigram: String = window.iter().collect();
                trigrams.push(trigram);
            }
            i += 1;
        }

        if trigrams.is_empty() && pattern.len() >= 3 {
            let consecutive_chars = self.find_longest_consecutive_chars(pattern);
            if consecutive_chars.len() >= 3 {
                let padded = format!("$${}", consecutive_chars);
                let chars: Vec<char> = padded.chars().collect();
                for i in 0..=chars.len().saturating_sub(3) {
                    let trigram: String = chars[i..i + 3].iter().collect();
                    trigrams.push(trigram);
                }
            }
        }

        trigrams
    }

    fn find_longest_consecutive_chars(&self, pattern: &str) -> String {
        let chars: Vec<char> = pattern.chars().collect();
        let mut longest = String::new();
        let mut current = String::new();

        for &ch in &chars {
            if ch != '*' && ch != '?' {
                current.push(ch);
            } else {
                if current.len() > longest.len() {
                    longest = current.clone();
                }
                current.clear();
            }
        }

        if current.len() > longest.len() {
            longest = current;
        }

        longest
    }

    fn matches_pattern(&self, term: &str, pattern: &str) -> bool {
        self.matches_wildcard(term, pattern)
    }

    fn matches_wildcard(&self, text: &str, pattern: &str) -> bool {
        let text_chars: Vec<char> = text.chars().collect();
        let pattern_chars: Vec<char> = pattern.chars().collect();

        self.matches_wildcard_recursive(&text_chars, &pattern_chars, 0, 0)
    }

    fn matches_wildcard_recursive(
        &self,
        text: &[char],
        pattern: &[char],
        text_idx: usize,
        pattern_idx: usize,
    ) -> bool {
        if pattern_idx >= pattern.len() {
            return text_idx >= text.len();
        }

        if text_idx >= text.len() {
            return pattern[pattern_idx..].iter().all(|&c| c == '*');
        }

        match pattern[pattern_idx] {
            '*' => {
                self.matches_wildcard_recursive(text, pattern, text_idx + 1, pattern_idx)
                    || self.matches_wildcard_recursive(text, pattern, text_idx, pattern_idx + 1)
            }
            '?' => self.matches_wildcard_recursive(text, pattern, text_idx + 1, pattern_idx + 1),
            ch => {
                if text[text_idx] == ch {
                    self.matches_wildcard_recursive(text, pattern, text_idx + 1, pattern_idx + 1)
                } else {
                    false
                }
            }
        }
    }

    pub fn memory_size(&self) -> usize {
        let mut size = std::mem::size_of::<TrigramIndex>();

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
    use crate::dictionary::{Dictionary, TermEntry};

    #[test]
    fn test_trigram_generation() {
        let trigrams = TrigramIndex::generate_trigrams_static("hello");
        assert!(trigrams.contains(&"$$h".to_string()));
        assert!(trigrams.contains(&"$he".to_string()));
        assert!(trigrams.contains(&"hel".to_string()));
        assert!(trigrams.contains(&"ell".to_string()));
        assert!(trigrams.contains(&"llo".to_string()));
    }

    #[test]
    fn test_trigram_search() {
        let mut dict = Dictionary::new();
        dict.terms.insert(
            "hello".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );
        dict.terms.insert(
            "help".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );
        dict.terms.insert(
            "world".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );

        let trigram_index = TrigramIndex::from_dictionary(&dict);

        let results = trigram_index.find_matching_terms("hel*");
        assert!(results.contains("hello"));
        assert!(results.contains("help"));
        assert!(!results.contains("world"));
    }

    #[test]
    fn test_trigram_wildcard() {
        let mut dict = Dictionary::new();
        dict.terms.insert(
            "testing".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );
        dict.terms.insert(
            "test".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );
        dict.terms.insert(
            "contest".to_string(),
            TermEntry {
                frequency: 1,
                documents: vec!["doc1".to_string()],
            },
        );

        let trigram_index = TrigramIndex::from_dictionary(&dict);

        let results = trigram_index.find_matching_terms("*est*");
        assert!(results.contains("testing"));
        assert!(results.contains("test"));
        assert!(results.contains("contest"));
    }
}
