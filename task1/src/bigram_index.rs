use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::dictionary::Dictionary;
use crate::query::{tokenize, QueryParser};

#[derive(Debug, Serialize, Deserialize)]
pub struct BigramIndex {
    pub index: HashMap<String, Vec<String>>,
    pub documents: Vec<String>,
}

impl BigramIndex {
    pub fn from_dictionary_with_parser<F>(
        dictionary: &Dictionary,
        file_parser: F,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        F: Fn(&str) -> Result<Vec<String>, Box<dyn std::error::Error>>,
    {
        println!("    BigramIndex: Starting index construction");
        let mut index = HashMap::new();
        let mut documents = HashSet::new();

        // Collect unique documents first to avoid duplicate processing
        println!("    BigramIndex: Collecting unique documents");
        for (_, term_entry) in &dictionary.terms {
            for document in &term_entry.documents {
                documents.insert(document.clone());
            }
        }
        println!("    BigramIndex: Found {} unique documents", documents.len());

        // Process each document only once
        println!("    BigramIndex: Processing documents");
        let mut processed_count = 0;
        for document in &documents {
            processed_count += 1;
            if processed_count % 10 == 0 {
                println!("    BigramIndex: Processed {}/{} documents", processed_count, documents.len());
            }
            
            let words = file_parser(document)?;
            
            let mut bigram_count = 0;
            for window in words.windows(2) {
                let bigram = format!("{} {}", window[0], window[1]);
                index.entry(bigram)
                    .or_insert_with(Vec::new)
                    .push(document.clone());
                bigram_count += 1;
            }
            
            if processed_count <= 5 || processed_count % 50 == 0 {
                println!("    BigramIndex: Document {} generated {} bigrams", document, bigram_count);
            }
        }

        println!("    BigramIndex: Deduplicating posting lists in parallel");
        index.par_iter_mut().for_each(|(_, posting_list)| {
            posting_list.sort();
            posting_list.dedup();
        });

        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();

        println!("    BigramIndex: Construction complete - {} bigrams, {} documents", index.len(), documents.len());
        Ok(BigramIndex { index, documents })
    }

    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.index.iter().map(|(k, v)| {
            k.len() + std::mem::size_of::<Vec<String>>() + 
            v.iter().map(|s| s.len()).sum::<usize>()
        }).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>()
    }

    pub fn search_phrase(&self, phrase: &str) -> Result<HashSet<String>, String> {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        if words.len() < 2 {
            return Err("Phrase must contain at least two words".to_string());
        }

        let mut result: Option<HashSet<String>> = None;

        for window in words.windows(2) {
            let bigram = format!("{} {}", window[0].to_lowercase(), window[1].to_lowercase());
            
            if let Some(docs) = self.index.get(&bigram) {
                let bigram_docs: HashSet<String> = docs.iter().cloned().collect();
                
                result = Some(match result {
                    None => bigram_docs,
                    Some(existing) => existing.intersection(&bigram_docs).cloned().collect(),
                });
            } else {
                return Ok(HashSet::new());
            }
        }

        Ok(result.unwrap_or_else(HashSet::new))
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
            
            if phrase_words.len() < 2 {
                return Err("Phrase must contain at least two words".to_string());
            }
            
            self.search_phrase(&phrase_words.join(" "))
        } else {
            let term = &tokens[*pos];
            *pos += 1;
            self.search_term(term)
        }
    }
}

impl QueryParser for BigramIndex {
    type Result = HashSet<String>;
    type Error = String;

    fn search(&self, query: &str) -> Result<Self::Result, Self::Error> {
        let tokens = tokenize(&query.to_lowercase())?;
        let mut pos = 0;
        self.parse_or_expr(&tokens, &mut pos)
    }

    fn search_term(&self, term: &str) -> Result<Self::Result, Self::Error> {
        Err(format!("Bigram index doesn't support single term search: '{}'", term))
    }
}