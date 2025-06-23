use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::dictionary::Dictionary;
use crate::query::{tokenize, QueryParser};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingEntry {
    pub document: String,
    pub positions: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CoordinateIndex {
    pub index: HashMap<String, Vec<PostingEntry>>,
    pub documents: Vec<String>,
}

impl CoordinateIndex {
    pub fn from_dictionary_with_parser<F>(
        dictionary: &Dictionary,
        file_parser: F,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        F: Fn(&str) -> Result<Vec<String>, Box<dyn std::error::Error>>,
    {
        println!("    CoordinateIndex: Starting index construction");
        let mut index: HashMap<String, HashMap<String, Vec<usize>>> = HashMap::new();
        let mut documents = HashSet::new();

        // Collect unique documents first to avoid duplicate processing
        println!("    CoordinateIndex: Collecting unique documents");
        for (_, term_entry) in &dictionary.terms {
            for document in &term_entry.documents {
                documents.insert(document.clone());
            }
        }
        println!("    CoordinateIndex: Found {} unique documents", documents.len());

        // Process each document only once
        println!("    CoordinateIndex: Processing documents");
        let mut processed_count = 0;
        for document in &documents {
            processed_count += 1;
            if processed_count % 10 == 0 {
                println!("    CoordinateIndex: Processed {}/{} documents", processed_count, documents.len());
            }
            
            let words = file_parser(document)?;
            
            for (position, word) in words.iter().enumerate() {
                index.entry(word.clone())
                    .or_insert_with(HashMap::new)
                    .entry(document.clone())
                    .or_insert_with(Vec::new)
                    .push(position);
            }
            
            if processed_count <= 5 || processed_count % 50 == 0 {
                println!("    CoordinateIndex: Document {} has {} words", document, words.len());
            }
        }

        println!("    CoordinateIndex: Converting to final format");
        let mut final_index = HashMap::new();
        let mut term_count = 0;
        for (term, doc_positions) in index {
            term_count += 1;
            if term_count % 1000 == 0 {
                println!("    CoordinateIndex: Processed {} terms", term_count);
            }
            
            let mut postings = Vec::new();
            for (document, positions) in doc_positions {
                postings.push(PostingEntry {
                    document,
                    positions,
                });
            }
            postings.sort_by(|a, b| a.document.cmp(&b.document));
            final_index.insert(term, postings);
        }

        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();

        println!("    CoordinateIndex: Construction complete - {} terms, {} documents", final_index.len(), documents.len());
        Ok(CoordinateIndex {
            index: final_index,
            documents,
        })
    }

    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.index.iter().map(|(k, v)| {
            k.len() + std::mem::size_of::<Vec<PostingEntry>>() + 
            v.iter().map(|entry| {
                entry.document.len() + 
                std::mem::size_of::<Vec<usize>>() + 
                entry.positions.len() * std::mem::size_of::<usize>()
            }).sum::<usize>()
        }).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>()
    }

    pub fn search_phrase(&self, phrase: &str) -> Result<HashSet<String>, String> {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        if words.is_empty() {
            return Ok(HashSet::new());
        }

        if words.len() == 1 {
            return self.search_term(words[0]);
        }

        let first_word = words[0].to_lowercase();
        let first_postings = match self.index.get(&first_word) {
            Some(postings) => postings,
            None => return Ok(HashSet::new()),
        };

        let mut result = HashSet::new();

        for posting in first_postings {
            let document = &posting.document;
            
            let mut all_words_found = true;
            let mut current_positions = posting.positions.clone();

            for (word_offset, word) in words.iter().enumerate().skip(1) {
                let word_lower = word.to_lowercase();
                
                if let Some(word_postings) = self.index.get(&word_lower) {
                    if let Some(word_posting) = word_postings.iter().find(|p| p.document == *document) {
                        let next_positions: Vec<usize> = current_positions.iter()
                            .filter_map(|&pos| {
                                if word_posting.positions.contains(&(pos + word_offset)) {
                                    Some(pos + word_offset)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if next_positions.is_empty() {
                            all_words_found = false;
                            break;
                        }
                        current_positions = next_positions;
                    } else {
                        all_words_found = false;
                        break;
                    }
                } else {
                    all_words_found = false;
                    break;
                }
            }

            if all_words_found && !current_positions.is_empty() {
                result.insert(document.clone());
            }
        }

        Ok(result)
    }

    pub fn search_proximity(&self, words: &[&str], max_distance: usize) -> Result<HashSet<String>, String> {
        if words.len() < 2 {
            return Err("Proximity search requires at least two words".to_string());
        }

        let first_word = words[0].to_lowercase();
        let first_postings = match self.index.get(&first_word) {
            Some(postings) => postings,
            None => return Ok(HashSet::new()),
        };

        let mut result = HashSet::new();

        for posting in first_postings {
            let document = &posting.document;
            let mut found_proximity = false;

            for &first_pos in &posting.positions {
                let mut all_words_in_range = true;

                for word in words.iter().skip(1) {
                    let word_lower = word.to_lowercase();
                    
                    if let Some(word_postings) = self.index.get(&word_lower) {
                        if let Some(word_posting) = word_postings.iter().find(|p| p.document == *document) {
                            let word_in_range = word_posting.positions.iter().any(|&pos| {
                                let distance = if pos > first_pos {
                                    pos - first_pos
                                } else {
                                    first_pos - pos
                                };
                                distance <= max_distance
                            });

                            if !word_in_range {
                                all_words_in_range = false;
                                break;
                            }
                        } else {
                            all_words_in_range = false;
                            break;
                        }
                    } else {
                        all_words_in_range = false;
                        break;
                    }
                }

                if all_words_in_range {
                    found_proximity = true;
                    break;
                }
            }

            if found_proximity {
                result.insert(document.clone());
            }
        }

        Ok(result)
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
        } else if tokens[*pos] == "\"" {
            *pos += 1;
            let mut phrase_words = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != "\"" {
                phrase_words.push(tokens[*pos].clone());
                *pos += 1;
            }
            if *pos >= tokens.len() {
                return Err("Missing closing quote".to_string());
            }
            *pos += 1;
            self.search_phrase(&phrase_words.join(" "))
        } else if tokens[*pos].starts_with("near/") {
            let distance_str = tokens[*pos].strip_prefix("near/").unwrap();
            let distance: usize = distance_str.parse()
                .map_err(|_| format!("Invalid distance in near operator: {}", distance_str))?;
            
            *pos += 1;
            if *pos >= tokens.len() || tokens[*pos] != "(" {
                return Err("Expected '(' after near operator".to_string());
            }
            *pos += 1;
            
            let mut words = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                words.push(tokens[*pos].as_str());
                *pos += 1;
            }
            if *pos >= tokens.len() {
                return Err("Missing closing parenthesis for near operator".to_string());
            }
            *pos += 1;
            
            if words.len() < 2 {
                return Err("Near operator requires at least two words".to_string());
            }
            
            self.search_proximity(&words, distance)
        } else {
            let term = &tokens[*pos];
            *pos += 1;
            self.search_term(term)
        }
    }
}

impl QueryParser for CoordinateIndex {
    type Result = HashSet<String>;
    type Error = String;

    fn search(&self, query: &str) -> Result<Self::Result, Self::Error> {
        let tokens = tokenize(&query.to_lowercase())?;
        let mut pos = 0;
        self.parse_or_expr(&tokens, &mut pos)
    }

    fn search_term(&self, term: &str) -> Result<Self::Result, Self::Error> {
        if let Some(postings) = self.index.get(term) {
            Ok(postings.iter().map(|p| p.document.clone()).collect())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}