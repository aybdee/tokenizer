use regex::Regex;
use std::collections::{HashMap, HashSet};
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

fn bpe_tokenizer<I: Iterator<Item = String>>(lines: I) {
    let mut characters: HashSet<char> = HashSet::new();
    let mut vocab: HashMap<VocabKey, usize> = HashMap::new();
    let mut corpus: HashMap<Vec<usize>, usize> = HashMap::new();
    let re = Regex::new(r"\w+|[^\w\s]+").unwrap();

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

            match corpus.get(&tokenized) {
                Some(_) => {
                    corpus.entry(tokenized).and_modify(|count| *count += 1);
                }
                None => {
                    corpus.insert(tokenized, 1);
                }
            }
        }
    }

    while vocab.len() < 50_000 {
        let mut pair_map: HashMap<(usize, usize), usize> = HashMap::new();
        for (word, count) in corpus.iter() {
            for (a, b) in word.iter().zip(word.iter().skip(1)) {
                *pair_map.entry((*a, *b)).or_insert(0) += count
            }
        }

        let Some((top_pair, top_pair_count)) = pair_map
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

        let new_index = vocab.len() + 1;
        vocab.insert(VocabKey::Merged(top_pair), new_index);

        corpus = corpus
            .into_iter()
            .map(|(word, count)| {
                let mut letters = vec![];
                let mut word_index = 0;
                while word_index < word.len() {
                    if word_index < word.len() - 1 {
                        let a = &word[word_index];
                        let b = &word[word_index + 1];
                        if *a == top_pair.0 && *b == top_pair.1 {
                            letters.push(new_index);
                            word_index += 2;
                            continue;
                        }
                    }
                    letters.push(word[word_index]);
                    word_index += 1;
                }

                (letters, count)
            })
            .collect::<HashMap<_, _>>();
    }

    println!("{}", vocab.len());
}

fn hf_baseline<I: Iterator<Item = String> + Send + Sync>(lines: I) {
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

    bpe_tokenizer(reader.lines().filter_map(Result::ok));
}
