use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer;

#[derive(Eq, PartialEq, Hash)]
enum VocabKey {
    Single(char),
    Merged((usize, usize)),
}

fn bpe_tokenizer<I: Iterator<Item = CompactString>>(lines: I) {
    let mut characters: AHashSet<char> = AHashSet::new();
    let mut vocab: AHashMap<VocabKey, usize> = AHashMap::new();
    let mut corpus_map: AHashMap<Vec<usize>, usize> = AHashMap::new();
    let re = Regex::new(r"\w+|[^\w\s]+").unwrap();
    let mut pair_counts: AHashMap<(usize, usize), usize> = AHashMap::new();

    for line in lines {
        for word in re.find_iter(&line).map(|m| m.as_str()) {
            let mut tokenized = vec![];
            for letter in word.chars() {
                let inserted = characters.insert(letter);
                if inserted {
                    let index = characters.len();
                    vocab.insert(VocabKey::Single(letter), index);
                    tokenized.push(index);
                } else {
                    tokenized.push(*vocab.get(&VocabKey::Single(letter)).unwrap());
                }
            }

            match corpus_map.get(&tokenized) {
                Some(_) => {
                    corpus_map.entry(tokenized).and_modify(|count| *count += 1);
                }
                None => {
                    corpus_map.insert(tokenized, 1);
                }
            }
        }
    }

    let mut corpus = corpus_map.into_iter().collect::<Vec<(_, _)>>();

    for (word, count) in corpus.iter() {
        for (a, b) in word.iter().zip(word.iter().skip(1)) {
            *pair_counts.entry((*a, *b)).or_insert(0) += count
        }
    }

    while vocab.len() < 50_000 {
        let Some((top_pair, top_pair_count)) = pair_counts
            .iter()
            .max_by(|(pa, ca), (pb, cb)| {
                ca.cmp(cb).then_with(|| pb.cmp(pa)) // to make ties deterministic
            })
            .map(|(k, v)| (*k, *v))
        else {
            break;
        };

        if top_pair_count < 2 {
            break;
        }

        let new_token = vocab.len() + 1;
        vocab.insert(VocabKey::Merged(top_pair), new_token);

        corpus = corpus
            .into_iter()
            .map(|(word, count)| {
                let mut letters = vec![];
                let mut word_index = 0;
                while word_index < word.len() {
                    if word_index + 1 < word.len() {
                        let a = &word[word_index];
                        let b = &word[word_index + 1];
                        if *a == top_pair.0 && *b == top_pair.1 {
                            //handle LHS of merge
                            if let Some(&prev) = letters.last() {
                                // Decrement count of broken pair: (prev, top_pair.0)
                                pair_counts
                                    .entry((prev, word[word_index]))
                                    .and_modify(|c| *c = c.saturating_sub(count));

                                // Increment the new pair: (prev, new_token)
                                *pair_counts.entry((prev, new_token)).or_insert(0) += count;
                            }

                            //decrement count of merged pair
                            pair_counts
                                .entry((word[word_index], word[word_index + 1]))
                                .and_modify(|counter| *counter = counter.saturating_sub(count));

                            //handle RHS of merge
                            if word_index + 2 < word.len() {
                                // Decrement count of broken pair: (top_pair.1, prev)
                                pair_counts
                                    .entry((word[word_index + 1], word[word_index + 2]))
                                    .and_modify(|counter| *counter = counter.saturating_sub(count));

                                // Increment the new pair: (new_token, prev)
                                *pair_counts
                                    .entry((new_token, word[word_index + 2]))
                                    .or_insert(0) += count;
                            }

                            letters.push(new_token);
                            word_index += 2;
                            continue;
                        }
                    }
                    letters.push(word[word_index]);
                    word_index += 1;
                }

                (letters, count)
            })
            .collect::<Vec<_>>();
    }

    println!("{}", vocab.len());
}

fn hf_baseline<I: Iterator<Item = CompactString> + Send + Sync>(lines: I) {
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(Whitespace));

    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .vocab_size(50_000)
        .min_frequency(2)
        .show_progress(true)
        .build()
        .into();

    tokenizer.train(&mut trainer, lines).unwrap();
    println!("{}", tokenizer.get_vocab_size(false));
}

fn main() {
    tokenizers::utils::parallelism::set_parallelism(false);
    let file = File::open("./data/text.txt").unwrap();
    let reader = BufReader::new(file);

    bpe_tokenizer(
        reader
            .lines()
            .filter_map(Result::ok)
            .map(CompactString::from),
    );
}
