# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import warnings
import zipfile

import gensim.downloader as api
import numpy as np
import requests
import torch

# from env import Decrypto
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)


# logging.basicConfig(level=logging.INFO)
def preprocess_word2vec(data_dir=None):
    pickle_path = f"{data_dir}/word2vec_embeddings.pickle"
    txt_path = f"{data_dir}/word2vec_vocabulary.txt"
    file_path = f"{data_dir}/word2vec_embeddings.npy"

    if os.path.exists(pickle_path):
        print(f"Loading pre-processed Word2Vec embeddings from {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    print("Loading Word2Vec model...")
    word2vec_model = api.load("word2vec-google-news-300")

    print("Processing Word2Vec embeddings...")
    vocabulary = []
    embeddings = []
    vocab_to_idx = {}
    index = 0
    for word in tqdm(
        word2vec_model.index_to_key, desc="Processing Word2Vec embeddings"
    ):
        embeddings.append(np.asarray(word2vec_model[word], dtype="float32"))
        vocabulary.append(word)
        vocab_to_idx[word] = index
        index += 1

    embeddings = np.stack(embeddings)
    print(f"Finished loading embeddings. Total entries: {len(embeddings)}")
    np.save(file_path, embeddings)
    print(f"Saving processed embeddings to {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(vocab_to_idx, f)

    print(f"Saving vocabulary to {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for word in vocabulary:
            f.write(f"{word}\n")

    print(f"Finished processing Word2Vec embeddings. Total entries: {len(embeddings)}")
    return embeddings


def load_glove_embeddings(file_path, data_dir=None):
    pickle_path = file_path + ".pickle"

    if os.path.exists(pickle_path):
        print(f"Loading pre-processed embeddings from {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(file_path):
        print("GloVe file not found. Checking for zip file...")
        url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
        zip_path = f"{data_dir}/glove.840B.300d.zip"

        if not os.path.exists(zip_path):
            print("Zip file not found. Downloading from Stanford NLP...")
            # Download the zip file
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with (
                open(zip_path, "wb") as f,
                tqdm(
                    desc="Downloading GloVe",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar,
            ):
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)
        else:
            print(f"Zip file found at {zip_path}")

        # Unzip the file
        print("Unzipping GloVe file...")
        extract_dir = os.path.dirname(file_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the extracted file
        print(os.listdir(extract_dir))
        extracted_file = next(
            (f for f in os.listdir(extract_dir) if f == "glove.840B.300d.txt"), None
        )
        if extracted_file:
            extracted_path = os.path.join(extract_dir, extracted_file)
            print(f"Found extracted file: {extracted_path}")
            if extracted_path != file_path:
                print(f"Renaming {extracted_path} to {file_path}")
                os.rename(extracted_path, file_path)
        else:
            print("Error: Could not find extracted .txt file")
            return None

        print(f"GloVe file extracted to {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} still does not exist after extraction")
        return None

    print(f"Processing embeddings from {file_path}")
    # embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    embeddings = []
    vocab_to_idx = {}
    with open(file_path, "r", encoding="utf-8") as f:
        index = 0
        for line in tqdm(f, total=total_lines, desc="Loading GloVe embeddings"):
            values = line.split()
            split_point = next(
                (
                    i
                    for i, v in enumerate(values)
                    if v.replace(".", "").replace("-", "").replace("e", "").isdigit()
                ),
                None,
            )

            if split_point is None:
                continue

            word = " ".join(values[:split_point])
            vector_values = values[split_point:]

            try:
                vector = np.asarray(vector_values, dtype="float32")
                if len(vector) != 300:
                    raise ValueError
                embeddings.append(vector)
                vocab_to_idx[word] = index
                index += 1
            except ValueError:
                continue
    embeddings = np.stack(embeddings)
    print(f"Finished loading embeddings. Total entries: {len(embeddings)}")
    np.save(file_path + ".npy", embeddings)
    print(f"Saving processed embeddings to {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(vocab_to_idx, f)


# Load baseline model and tokenizer
def load_model_and_tokenizer(model_name, data_dir=None):
    os.makedirs(data_dir, exist_ok=True)
    if model_name.lower() == "glove":
        # global glove_embeddings
        load_glove_embeddings(f"{data_dir}/glove.840B.300d.txt", data_dir)

    elif model_name.lower() == "word2vec":
        # global word2vec_embeddings
        preprocess_word2vec(data_dir)


def cosine_similarity(a, b, word_a=None, word_b=None):
    with warnings.catch_warnings(record=True) as w:  # noqa F841
        warnings.simplefilter("always")
        result = dot(a, b) / (norm(a) * norm(b))
    return result


def get_top_k_closest_words(target_embedding, word_embeddings, k, target_word):
    similarities = []
    for word, embedding in word_embeddings.items():
        similarity = cosine_similarity(target_embedding, embedding, target_word, word)
        similarities.append((word, similarity))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_similarities[:k]]


class EmbeddingBaseline:
    def __init__(
        self,
        role,
        hint_words=None,
        smart_encoder=False,
        no_repeat_hints=True,
        allow_target_keyword=False,
        seed=0,
        k=128,
        model_name=None,
        model=None,
        tokenizer=None,
        model_embeddings=None,
        global_guess=False,
        vocab_to_idx=None,
    ):
        self.role = role
        self.smart_encoder = smart_encoder
        self.no_repeat_hints = no_repeat_hints
        self.allow_target_keyword = allow_target_keyword
        self.used_hints = {
            0: set(),
            1: set(),
            2: set(),
            3: set(),
        }  # To store used hints for each keyword position
        self.k_counter = {0: k, 1: k, 2: k, 3: k}
        self.hint_words = hint_words
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_embeddings = model_embeddings
        self.k = k
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.global_guess = global_guess
        self.vocab_to_idx = vocab_to_idx
        if hint_words:
            self.hint_embeddings = self.precompute_hint_embeddings(hint_words)

    def get_embedding(self, word):
        if self.model_name.lower() in ["glove", "word2vec"]:
            if word in self.vocab_to_idx.keys():
                idx = self.vocab_to_idx[word]
                return self.model_embeddings[idx]
            else:
                return np.ones(300)
        else:
            inputs = self.tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def precompute_hint_embeddings(self, hint_words):
        # global word_embeddings
        word_embeddings = {}
        for word in hint_words:
            word_embeddings[word] = self.get_embedding(word)
        return word_embeddings

    def _simple_encoder_hint(self, code, keywords):
        code = int(code)
        keyword = keywords[code - 1]
        keyword_embedding = self.get_embedding(keyword)
        closest_words = get_top_k_closest_words(
            keyword_embedding, self.hint_embeddings, self.k_counter[code - 1], keyword
        )
        print("BEFORE", self.k_counter)
        self.k_counter[code - 1] += 1
        print("AFTER", self.k_counter)
        if self.no_repeat_hints:
            unused_words = [
                word for word in closest_words if word not in self.used_hints[code - 1]
            ]
        else:
            unused_words = closest_words
        if unused_words:
            hint = self.rng.choice(unused_words)
        else:
            print("FALLBACK")
            hint = self._fallback_hint(code, keyword)
        if self.no_repeat_hints:
            self.used_hints[code - 1].add(hint)
        return hint

    def _smart_encoder_hint(self, code, keywords):
        code = int(code)
        target_keyword = keywords[code - 1]
        target_embedding = self.get_embedding(target_keyword)
        other_keywords = [kw for i, kw in enumerate(keywords) if i != code - 1]
        other_embeddings = [self.get_embedding(kw) for kw in other_keywords]

        candidates = get_top_k_closest_words(
            target_embedding,
            self.hint_embeddings,
            self.k_counter[code - 1],
            target_keyword,
        )
        self.k_counter[code - 1] += 1
        if not self.allow_target_keyword:
            candidates = [
                word for word in candidates if word.lower() != target_keyword.lower()
            ]
        if self.no_repeat_hints:
            candidates = [
                word for word in candidates if word not in self.used_hints[code - 1]
            ]
        self.rng.shuffle(candidates)

        # Check similarity with previously chosen hints
        for candidate in candidates:
            candidate_embedding = self.get_embedding(candidate)
            target_similarity = cosine_similarity(
                candidate_embedding, target_embedding, candidate, target_keyword
            )

            # Check similarity with other keywords
            other_similarities = [
                cosine_similarity(candidate_embedding, emb, candidate, kw)
                for emb, kw in zip(other_embeddings, other_keywords)
            ]

            # Combine all other similarities
            all_other_similarities = other_similarities
            if target_similarity > max(all_other_similarities):
                if self.no_repeat_hints:
                    self.used_hints[code - 1].add(candidate)
                return candidate

        # If no suitable candidate is found, fall back to the fallback hint method
        print("FALLBACK")
        return self._fallback_hint(code, target_keyword)

    def encoder_hint(self, code, keywords):
        if not self.smart_encoder:
            return [self._simple_encoder_hint(c, keywords) for c in code]
        else:
            return [self._smart_encoder_hint(c, keywords) for c in code]

    def _fallback_hint(self, code, target_keyword):
        target_embedding = self.get_embedding(target_keyword)
        candidates = get_top_k_closest_words(
            target_embedding, self.hint_embeddings, self.k, target_keyword
        )

        if not self.allow_target_keyword:
            candidates = [
                word for word in candidates if word.lower() != target_keyword.lower()
            ]

        if self.no_repeat_hints:
            candidates = [
                word for word in candidates if word not in self.used_hints[code - 1]
            ]

        if candidates:
            fallback_hint = self.rng.choice(candidates)
        else:
            # If no suitable candidates are found, choose randomly from common words
            fallback_hint = self.rng.choice(
                [
                    word
                    for word in self.hint_words
                    if word.lower() != target_keyword.lower()
                ]
            )

        if self.no_repeat_hints:
            self.used_hints[code - 1].add(fallback_hint)

        return fallback_hint

    def reset_used_hints(self):
        if self.no_repeat_hints:
            self.used_hints = {0: set(), 1: set(), 2: set(), 3: set()}
        self.k_counter = {0: self.k, 1: self.k, 2: self.k, 3: self.k}

    def decoder_greedy_guess(self, hints, keywords, hint_history):
        """
        Greedily match each hint to its most similar keyword and return the corresponding guess.
        """
        guesses = []
        used_indices = set()

        for hint in hints:
            hint_embedding = self.get_embedding(hint)
            similarities = []
            for i, (keyword, history) in enumerate(zip(keywords, hint_history)):
                keyword_embedding = self.get_embedding(keyword)
                avg_embedding = keyword_embedding
                similarity = cosine_similarity(
                    hint_embedding, avg_embedding, hint, keyword
                )
                similarities.append((i + 1, similarity))

            # Sort similarities and choose the highest unused index
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            for index, _ in sorted_similarities:
                if index not in used_indices:
                    guesses.append(index)
                    used_indices.add(index)
                    break

        return guesses

    def decoder_global_guess(self, hints, keywords, hint_history):
        """
        Compute the similarity-maximizing assignment of hints to keywords and return the corresponding guess.
        """
        # Create a similarity matrix
        similarity_matrix = np.zeros((len(hints), len(keywords)))

        for i, hint in enumerate(hints):
            hint_embedding = self.get_embedding(hint)
            for j, (keyword, history) in enumerate(zip(keywords, hint_history)):
                keyword_embedding = self.get_embedding(keyword)
                avg_embedding = keyword_embedding
                similarity = cosine_similarity(
                    hint_embedding, avg_embedding, hint, keyword
                )
                similarity_matrix[i, j] = similarity

        # Use the Hungarian algorithm to find the optimal assignment
        # We use negative similarities because the algorithm minimizes the cost
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

        # Create the guess based on the optimal assignment
        guess = [0] * len(hints)
        for hint_index, keyword_index in zip(row_indices, col_indices):
            guess[hint_index] = (
                keyword_index + 1
            )  # Add 1 because code indices start at 1

        return guess

    def decoder_guess(self, hints, keywords, hint_history):
        if self.global_guess:
            return self.decoder_global_guess(hints, keywords, hint_history)
        else:
            return self.decoder_greedy_guess(hints, keywords, hint_history)

    def interceptor_greedy_guess(self, hints, hint_history):
        """
        Greedily match each hint to its most similar hint history and return the corresponding guess.
        """
        guess = []
        used_indices = set()

        for hint in hints:
            hint_embedding = self.get_embedding(hint)
            hint_similarities = []
            for history in hint_history:
                if history:
                    history_embeddings = [self.get_embedding(h) for h in history]
                    avg_history_embedding = np.mean(history_embeddings, axis=0)
                    similarity = cosine_similarity(
                        hint_embedding, avg_history_embedding
                    )
                else:
                    similarity = 0  # No history for this keyword yet
                hint_similarities.append(similarity)

            # Find the highest similarity that hasn't been used yet
            sorted_indices = np.argsort(hint_similarities)[::-1]
            for index in sorted_indices:
                if index + 1 not in used_indices:
                    guess.append(index + 1)
                    used_indices.add(index + 1)
                    break

        return guess

    def interceptor_global_guess(self, hints, hint_history):
        """
        Compute the similarity-maximizing assignment of hints to histories and return the corresponding guess.
        """
        # Create a similarity matrix
        similarity_matrix = np.zeros((len(hints), len(hint_history)))

        for i, hint in enumerate(hints):
            hint_embedding = self.get_embedding(hint)
            for j, history in enumerate(hint_history):
                if history:
                    history_embeddings = [self.get_embedding(h) for h in history]
                    avg_history_embedding = np.mean(history_embeddings, axis=0)
                    similarity = cosine_similarity(
                        hint_embedding, avg_history_embedding, hint, f"history_{j}"
                    )
                else:
                    similarity = 0  # No history for this keyword yet
                similarity_matrix[i, j] = similarity

        # Use the Hungarian algorithm to find the optimal assignment
        # We use negative similarities because the algorithm minimizes the cost
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

        # Create the guess based on the optimal assignment
        guess = [0] * len(hints)
        for hint_index, history_index in zip(row_indices, col_indices):
            guess[hint_index] = (
                history_index + 1
            )  # Add 1 because code indices start at 1

        return guess

    def interceptor_guess(self, hints, hint_history):
        if self.global_guess:
            return self.interceptor_global_guess(hints, hint_history)
        else:
            return self.interceptor_greedy_guess(hints, hint_history)
