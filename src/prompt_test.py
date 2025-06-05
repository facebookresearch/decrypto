# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from env import (
    Decrypto,
    format_code,
    get_decoder_prompt,
    get_encoder_prompt,
    get_interceptor_prompt,
)


def run_dummy_game(args):
    env = Decrypto()
    code, keywords, info = env.reset(seed=1789)

    prev_interceptions = 0
    prev_miscommunications = 0

    for step in range(args.num_steps):
        print("\n\n")

        encoder_prompt = get_encoder_prompt(
            info, keywords, code, mode=args.encoder_mode
        )
        hints = ["hint_one", "hint_two", "hint_three"]

        decoder_prompt = get_decoder_prompt(
            keywords, info, hints, mode=args.decoder_mode
        )
        guess_bob = [1, 2, 3]

        interceptor_prompt = get_interceptor_prompt(
            info, hints, mode=args.interceptor_mode
        )
        guess_eve = [4, 3, 2]
        if step == 1:
            guess_eve = [int(code[0]), int(code[1]), int(code[2])]

        new_code, rewards, done, info = env.step(hints, guess_bob, guess_eve)

        print(encoder_prompt + "\n +++++++++++++++++\n")
        print(decoder_prompt + "\n +++++++++++++++++\n")
        print(interceptor_prompt + "\n ==================================")

        turn_summary = f"""Turn {info["turn"]} summary:
    Code : {format_code(code)}
    Hints : {format_code(hints)}
    Decoder guess : {format_code(guess_bob)}
    Interceptor guess : {format_code(guess_eve)}
"""

        if info["miscommunications"] > prev_miscommunications:
            prev_miscommunications = info["miscommunications"]
            turn_summary += "    The Decoder failed to guess the code. The Encoder-Decoder team get a Miscommunication token.\n"
        if info["interceptions"] > prev_interceptions:
            prev_interceptions = info["interceptions"]
            turn_summary += "    The Interceptor successfully guessed the code and gained an Interception token.\n"

        info["prev_turn_summary"] = turn_summary

        code = new_code

        input("\n >>> Press Enter to continue.")
        os.system("clear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_mode", type=int, default=0)
    parser.add_argument("--decoder_mode", type=int, default=0)
    parser.add_argument("--interceptor_mode", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=3)

    run_dummy_game(parser.parse_args())
