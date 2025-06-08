use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::dictionary::Dictionary;
use crate::query::{tokenize, QueryParser};

#[derive(Debug, Serialize, Deserialize)]
pub struct InvertedIndex {
    pub index: HashMap<String, Vec<String>>,
    pub documents: Vec<String>,
}

impl InvertedIndex {
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        let mut index = HashMap::new();
        let mut documents = HashSet::new();
        
        for (term, term_entry) in &dictionary.terms {
            index.insert(term.clone(), term_entry.documents.clone());
            for doc in &term_entry.documents {
                documents.insert(doc.clone());
            }
        }
        
        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();
        
        InvertedIndex { index, documents }
    }
    
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.index.iter().map(|(k, v)| {
            k.len() + std::mem::size_of::<Vec<String>>() + 
            v.iter().map(|s| s.len()).sum::<usize>()
        }).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>()
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
            Ok(docs.iter().cloned().collect())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}