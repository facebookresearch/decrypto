# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np


def load_wordlist(filename="decrypto_words.txt"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    with open(file_path, "r") as file:
        return [word.strip() for word in file.readlines()]


WORDLIST = load_wordlist()


class Decrypto:
    def __init__(self, max_turns=8):
        self.max_turns = max_turns
        self.interceptions = None
        self.keywords = None
        self.miscommunications = None
        self.turn = None
        self.seed = None
        self.rng = None
        self.current_code = None
        self.done = False
        self.code_history = []
        self.hint_history = [[], [], [], []]

    def reset(self, seed, keywords=None):
        """
        Start a new game with new keywords
        :param seed: experiments eed
        :param keywords: (Optional) List of 4 custom keywords
        :return: first code (str), keywords (List[str]), info (Dict)
        """

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        if keywords is not None:
            self.keywords = keywords
        else:
            word_bank = WORDLIST
            self.keywords = list(self.rng.choice(word_bank, 4, replace=False))

        self.interceptions = 0
        self.miscommunications = 0
        self.turn = 0
        self.done = False
        self.code_history = []
        self.hint_history = [[], [], [], []]

        code = self.rng.choice(np.arange(1, 5), size=3, replace=False)
        self.current_code = code
        code_str = self.code_to_string(code)
        self.code_history.append(code)

        info = {
            "turn": self.turn,
            "miscommunications": self.miscommunications,
            "interceptions": self.interceptions,
            "code history": self.code_history[:-1],
            "hint history": self.hint_history,
        }

        return code_str, self.keywords, info

    def step(self, hints, guess_bob, guess_eve):
        """
        Perform one turn of the game. Even turns are Alice, Odd turns are Bob/Eve

        :param hints: list of hints provided by Alice
        :param guess_bob: Bob's guess, as array
        :param guess_eve: Eve's guess, as array
        :return: observation, rewards, done, info
        """

        # Track hints
        for digit, hint in zip(self.current_code, hints):
            self.hint_history[digit - 1].append(hint)

        # Verify guesses
        if not (guess_bob == self.current_code).all():
            self.miscommunications += 1
        if (guess_eve == self.current_code).all():
            self.interceptions += 1

        # Sample new, unseen code
        code = self.rng.choice(np.arange(1, 5), size=3, replace=False)
        while any([np.array_equal(old_code, code) for old_code in self.code_history]):
            code = self.rng.choice(np.arange(1, 5), size=3, replace=False)
        code_str = self.code_to_string(code)
        self.current_code = code
        self.code_history.append(code)

        self.turn += 1

        rewards = {"alice": 0, "bob": 0, "eve": 0}

        if self.miscommunications == 2 or self.interceptions == 2:
            # Eve wins
            self.done = True
            rewards = {"alice": -1, "bob": -1, "eve": 1}
        elif self.turn == self.max_turns:
            # Alice and Bob win
            self.done = True
            rewards = {"alice": 1, "bob": 1, "eve": -1}

        # Info only contains public information
        info = {
            "turn": self.turn,
            "miscommunications": self.miscommunications,
            "interceptions": self.interceptions,
            "code history": self.code_history[:-1],
            "hint history": self.hint_history,
        }

        return code_str, rewards, self.done, info

    def code_to_string(self, code):
        return str(code[0]) + str(code[1]) + str(code[2])
