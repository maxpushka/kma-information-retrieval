use crate::CompressedDictionary;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuffixNode {
    pub children: HashMap<char, Box<SuffixNode>>,
    pub is_terminal: bool,
    pub terms: HashSet<String>,
}

impl SuffixNode {
    fn new() -> Self {
        SuffixNode {
            children: HashMap::new(),
            is_terminal: false,
            terms: HashSet::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SuffixTree {
    root: SuffixNode,
}

impl SuffixTree {
    pub fn new() -> Self {
        SuffixTree {
            root: SuffixNode::new(),
        }
    }

    pub fn from_dictionary(dictionary: &CompressedDictionary) -> Self {
        println!(
            "      SuffixTree: Processing {} terms in parallel",
            dictionary.sorted_terms.len()
        );

        let tree = Arc::new(Mutex::new(SuffixTree::new()));
        let terms = dictionary.extract_terms_parallel();

        // Process terms in parallel chunks for better progress reporting
        let chunk_size = 1000;
        let chunks: Vec<_> = terms.chunks(chunk_size).collect();

        chunks
            .par_iter()
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let local_tree = Arc::clone(&tree);

                for term in *chunk {
                    if let Ok(mut tree_lock) = local_tree.lock() {
                        tree_lock.add_term(term);
                    }
                }

                let processed = (chunk_idx + 1) * chunk_size;
                if processed % 5000 == 0 || chunk_idx == chunks.len() - 1 {
                    println!(
                        "      SuffixTree: Processed ~{} terms",
                        processed.min(terms.len())
                    );
                }
            });

        println!(
            "      SuffixTree: Complete - {} terms processed",
            terms.len()
        );

        // Extract the tree from Arc<Mutex<>>
        Arc::try_unwrap(tree).unwrap().into_inner().unwrap()
    }
    
    pub fn from_compressed_dictionary(dictionary: &CompressedDictionary) -> Self {
        println!(
            "      SuffixTree: Processing {} terms in parallel",
            dictionary.sorted_terms.len()
        );

        let tree = Arc::new(Mutex::new(SuffixTree::new()));
        let terms = dictionary.extract_terms_parallel();

        // Process terms in parallel chunks for better progress reporting
        let chunk_size = 1000;
        let chunks: Vec<_> = terms.chunks(chunk_size).collect();

        chunks.par_iter().enumerate().for_each(|(chunk_idx, chunk)| {
            // Build local suffix trees for this chunk
            let mut local_tree = SuffixTree::new();
            for term in chunk.iter() {
                local_tree.add_term(term);
            }

            // Merge local tree into global tree
            if let Ok(mut global_tree) = tree.lock() {
                Self::merge_trees(&mut global_tree.root, &local_tree.root);
            }

            let processed = (chunk_idx + 1) * chunk_size;
            if processed % 5000 == 0 || chunk_idx == chunks.len() - 1 {
                println!(
                    "      SuffixTree: Processed ~{} terms",
                    processed.min(terms.len())
                );
            }
        });

        println!(
            "      SuffixTree: Complete - {} terms processed",
            terms.len()
        );

        // Extract the tree from Arc<Mutex<>>
        Arc::try_unwrap(tree).unwrap().into_inner().unwrap()
    }

    fn add_term(&mut self, term: &str) {
        let term_chars: Vec<char> = term.chars().collect();

        for start_idx in 0..term_chars.len() {
            let suffix = &term_chars[start_idx..];
            self.insert_suffix(suffix, term);
        }
    }

    fn insert_suffix(&mut self, suffix: &[char], original_term: &str) {
        let mut current = &mut self.root;

        for &ch in suffix {
            current = current
                .children
                .entry(ch)
                .or_insert_with(|| Box::new(SuffixNode::new()));
            current.terms.insert(original_term.to_string());
        }

        current.is_terminal = true;
    }
    
    fn merge_trees(global: &mut SuffixNode, local: &SuffixNode) {
        // Merge terms
        global.terms.extend(local.terms.iter().cloned());
        global.is_terminal = global.is_terminal || local.is_terminal;
        
        // Merge children
        for (ch, local_child) in &local.children {
            let global_child = global.children
                .entry(*ch)
                .or_insert_with(|| Box::new(SuffixNode::new()));
            Self::merge_trees(global_child, local_child);
        }
    }

    pub fn find_matching_terms(&self, pattern: &str) -> HashSet<String> {
        if pattern.is_empty() {
            return HashSet::new();
        }

        let mut results = HashSet::new();
        self.find_with_wildcards(&self.root, pattern, &mut results);
        results
    }

    fn find_with_wildcards(&self, node: &SuffixNode, pattern: &str, results: &mut HashSet<String>) {
        if pattern.is_empty() {
            results.extend(node.terms.iter().cloned());
            return;
        }

        let chars: Vec<char> = pattern.chars().collect();
        let first_char = chars[0];
        let remaining = &pattern[1..];

        if first_char == '*' {
            results.extend(node.terms.iter().cloned());

            for child in node.children.values() {
                self.find_with_wildcards(child, pattern, results);
                self.find_with_wildcards(child, remaining, results);
            }
        } else if first_char == '?' {
            for child in node.children.values() {
                self.find_with_wildcards(child, remaining, results);
            }
        } else {
            if let Some(child) = node.children.get(&first_char) {
                self.find_with_wildcards(child, remaining, results);
            }
        }
    }

    pub fn memory_size(&self) -> usize {
        self.calculate_node_size(&self.root)
    }

    fn calculate_node_size(&self, node: &SuffixNode) -> usize {
        let mut size = std::mem::size_of::<SuffixNode>();

        size += node.children.len() * std::mem::size_of::<(char, Box<SuffixNode>)>();
        for child in node.children.values() {
            size += self.calculate_node_size(child);
        }

        size += node.terms.len() * std::mem::size_of::<String>();
        for term in &node.terms {
            size += term.len();
        }

        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::{Dictionary, CompressedDictionary};

    #[test]
    fn test_suffix_tree_basic() {
        let mut dict = Dictionary::new();
        dict.add_term("cat".to_string(), "doc1".to_string());
        dict.add_term("car".to_string(), "doc1".to_string());
        dict.add_term("card".to_string(), "doc1".to_string());

        let compressed_dict = CompressedDictionary::from_dictionary(&dict);
        let tree = SuffixTree::from_dictionary(&compressed_dict);

        let results = tree.find_matching_terms("ca*");
        assert!(results.contains("cat"));
        assert!(results.contains("car"));
        assert!(results.contains("card"));

        let results = tree.find_matching_terms("car");
        assert!(results.contains("car"));
        assert!(results.contains("card"));
    }

    #[test]
    fn test_wildcard_queries() {
        let mut dict = Dictionary::new();
        dict.add_term("test".to_string(), "doc1".to_string());
        dict.add_term("testing".to_string(), "doc1".to_string());
        dict.add_term("tester".to_string(), "doc1".to_string());

        let compressed_dict = CompressedDictionary::from_dictionary(&dict);
        let tree = SuffixTree::from_dictionary(&compressed_dict);

        let results = tree.find_matching_terms("test*");
        assert_eq!(results.len(), 3);

        let results = tree.find_matching_terms("test??");
        assert!(results.contains("tester"));

        let results = tree.find_matching_terms("*ing");
        assert!(results.contains("testing"));
    }
}
