# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import itertools
import json
import os
import pickle as pkl
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agents.embedding_baseline import load_model_and_tokenizer
from src.agents.role_client import RoleClient
from src.env import (
    Decrypto,
    format_code,
    get_decoder_prompt,
    get_encoder_prompt,
    get_gopnik_prompt,
    get_interceptor_prompt,
    get_prediction_prompt,
    get_system_prompt,
    get_tom_prompt,
)
from src.types import BaselineModel, Human, LocalModel
from src.utils.json_utils import compare_kw_lists
from src.utils.server import get_available_servers


def combo_to_string(combo):
    combo_string = ""
    for i in range(3):
        combo_string += f"({combo[i][0].model_key},{combo[i][1]},{combo[i][2]}), "
    combo_string += f"env_seed {combo[3]}"

    return combo_string


def run_games(
    encoder_policy,
    decoder_policy,
    interceptor_policy,
    env_seed=0,
    args=None,
):
    """
    Run Decrypto games with the given policies and environment seed. Can also run ToM experiments (Piaget, Gopnik).
    :param encoder_policy: Tuple of (model, system_mode, role_mode, tom_mode)
    :param decoder_policy: Tuple of (model, system_mode, role_mode)
    :param interceptor_policy: Tuple of (model, system_mode, role_mode)
    :param env_seed: Seed for the environment
    :param args: Configuration arguments
    :return: Dictionary containing game statistics and histories
    """
    encoder_model, encoder_system_mode, encoder_role_mode, encoder_tom_mode = (
        encoder_policy
    )
    decoder_model, decoder_system_mode, decoder_role_mode = decoder_policy
    interceptor_model, interceptor_system_mode, interceptor_role_mode = (
        interceptor_policy
    )
    all_histories = []

    encoder_client = RoleClient(
        "encoder",
        encoder_model,
        mode=encoder_system_mode,
        role_mode=encoder_role_mode,
        tom_mode=encoder_tom_mode,
        baseline_data_dir=args.baseline_data_dir,
        no_error_history=args.no_error_history,
    )

    decoder_client = RoleClient(
        "decoder",
        decoder_model,
        mode=decoder_system_mode,
        role_mode=decoder_role_mode,
        baseline_data_dir=args.baseline_data_dir,
        no_error_history=args.no_error_history,
    )

    interceptor_client = RoleClient(
        "interceptor",
        interceptor_model,
        mode=interceptor_system_mode,
        role_mode=interceptor_role_mode,
        baseline_data_dir=args.baseline_data_dir,
        no_error_history=args.no_error_history,
    )

    end_cond = {"intercept": 0, "miscomm": 0, "both": 0, "survived": 0}
    total_turns = 0
    intercept_turns = []
    miscomm_turns = []
    changed_hints = []
    encoder_attempts = []
    decoder_attempts = []
    interceptor_attempts = []
    total_fails = {
        "encoder": 0,
        "decoder": 0,
        "interceptor": 0,
    }
    alice_prediction_success = []
    alice_prediction_success_on_intercept = []
    alice_prediction_is_code = []

    gopnik_tom = {
        "weak_rep_change": [],
        "weak_false_belief": [],
        "self_other_consistency": [],
        "strong_rep_change": [],
        "strong_false_belief": [],
    }

    for ep in range(args.num_episodes):
        env = Decrypto()
        ep_seed = ep + env_seed * args.num_episodes
        code, keywords, info = env.reset(seed=ep_seed)
        encoder_client.reset_used_hints()  # Reset hints at the start of each episode

        combined_history = []
        encoder_client.reset_history()
        decoder_client.reset_history()
        interceptor_client.reset_history()

        episode_turns = 0
        prev_interceptions = 0
        num_interceptions = 0
        prev_miscommunications = 0
        changed_hint = 0
        alice_predicted = 0
        alice_predicted_on_intercept = 0
        alice_predicted_code = 0
        intercepted = False
        miscommed = False
        survived = False
        both = False
        per_episode_intercept_turns = []
        per_episode_miscomm_turns = []

        gopnik_ep_metrics = {
            "num_failed_kw_pred": 0,
            "pass_weak_RP": 0,  # Representational Change
            "pass_weak_FB": 0,  # False belief
            "pass_SO_consistency": 0,  # Self-Other consistency
            "pass_strong_RP": 0,
            "pass_strong_FB": 0,
        }

        for step in range(8):
            # 1. Encoder (Alice) provides hints
            encoder_prompt = get_encoder_prompt(
                info, keywords, code, mode=encoder_role_mode
            )

            if info["turn"] == 0 and encoder_client.pass_system_as_user:
                encoder_prompt = encoder_client.system_prompt + encoder_prompt

            combined_history.append(
                {"role": "user", "content": f"[ENCODER] {encoder_prompt}"}
            )

            hints, encoder_output, encoder_info = encoder_client.chat_completion(
                encoder_prompt, code, keywords, None, None
            )
            encoder_attempts.append(encoder_info["attempts"])
            total_fails["encoder"] += encoder_info["fail"]
            if hints is None:
                print("Error: Unable to get valid encoder output")
                continue

            combined_history.append(
                {"role": "assistant", "content": f"[ENCODER] {encoder_output}"}
            )

            # 1.1 (ToM experiment) Test if Alice can predict Eve's guess.
            if args.piaget and not isinstance(encoder_model, BaselineModel):
                prediction_prompt = get_prediction_prompt(info, keywords, code, hints)
                combined_history.append(
                    {"role": "user", "content": f"[ENCODER] {prediction_prompt}"}
                )
                prediction_alice, encoder_output, encoder_info = (
                    encoder_client.chat_completion(
                        prediction_prompt,
                        None,
                        keywords,
                        hints,
                        info["hint history"],
                        predict_code=True,
                    )
                )
                if prediction_alice is None:
                    print("Error: Unable to get valid encoder prediction output")
                    continue
                combined_history.append(
                    {"role": "assistant", "content": f"[ENCODER] {encoder_output}"}
                )

            # 1.2 Prompt Alice to reflect about other players and change hints if needed (not in paper)
            if args.theory_of_mind and not isinstance(encoder_model, BaselineModel):
                tom_prompt = get_tom_prompt(
                    info, keywords, code, hints, mode=encoder_tom_mode
                )
                combined_history.append(
                    {"role": "user", "content": f"[ENCODER] {tom_prompt}"}
                )
                hints_tom, encoder_output, encoder_info = (
                    encoder_client.chat_completion(
                        tom_prompt, code, keywords, None, None
                    )
                )
                encoder_attempts.append(encoder_info["attempts"])
                total_fails["encoder"] += encoder_info["fail"]
                if hints_tom is None:
                    print("Error: Unable to get valid encoder ToM output")
                    continue
                if hints != hints_tom:
                    changed_hint += 1
                hints = hints_tom
                combined_history.append(
                    {"role": "assistant", "content": f"[ENCODER] {encoder_output}"}
                )

            # 2. Decoder (Bob) makes a guess
            decoder_prompt = get_decoder_prompt(
                keywords, info, hints, mode=decoder_role_mode
            )

            if info["turn"] == 0 and decoder_client.pass_system_as_user:
                decoder_prompt = decoder_client.system_prompt + decoder_prompt

            combined_history.append(
                {"role": "user", "content": f"[DECODER] {decoder_prompt}"}
            )

            guess_bob, decoder_output, decoder_info = decoder_client.chat_completion(
                decoder_prompt, None, keywords, hints, info["hint history"]
            )
            decoder_attempts.append(decoder_info["attempts"])
            total_fails["decoder"] += decoder_info["fail"]
            if guess_bob is None:
                print("Error: Unable to get valid decoder output")
                continue

            combined_history.append(
                {"role": "assistant", "content": f"[DECODER] {decoder_output}"}
            )

            # 3.0 (ToM experiment) Test Eve's representational change and false belief abilities
            if (
                args.gopnik
                and not isinstance(interceptor_model, BaselineModel)
                and step != 0
            ):
                gopnik_predictions = {}

                for pred_question in ["vanilla", "rep_change", "false_belief"]:
                    prediction_prompt = get_gopnik_prompt(info, keywords, pred_question)

                    combined_history.append(
                        {
                            "role": "user",
                            "content": f"[INTERCEPTOR] {prediction_prompt}",
                        }
                    )
                    prediction_eve, interceptor_output, interceptor_info = (
                        interceptor_client.chat_completion(
                            prediction_prompt,
                            None,
                            None,
                            None,
                            info["hint history"],
                            predict_keywords=True,
                        )
                    )
                    if prediction_eve is None:
                        print("Error: Unable to get valid encoder prediction output")
                        gopnik_predictions = {}
                        break

                    gopnik_predictions[pred_question] = prediction_eve.copy()

                    combined_history.append(
                        {
                            "role": "assistant",
                            "content": f"[INTERCEPTOR] {interceptor_output}",
                        }
                    )

                if len(gopnik_predictions) == 3:
                    if not compare_kw_lists(gopnik_predictions["vanilla"], keywords):
                        # Only look at instances where agent does not identify the keywords
                        gopnik_ep_metrics["num_failed_kw_pred"] += 1

                        # LLMs pass only if their own statement is consistent with "vanilla" belief before reveal
                        gopnik_ep_metrics["pass_strong_RP"] += compare_kw_lists(
                            gopnik_predictions["vanilla"],
                            gopnik_predictions["rep_change"],
                        )
                        gopnik_ep_metrics["pass_strong_FB"] += compare_kw_lists(
                            gopnik_predictions["vanilla"],
                            gopnik_predictions["false_belief"],
                        )
                        gopnik_ep_metrics["pass_SO_consistency"] += compare_kw_lists(
                            gopnik_predictions["rep_change"],
                            gopnik_predictions["false_belief"],
                        )

                        # LLMs pass as long as they don't assign Past/Other belief of true keywords
                        gopnik_ep_metrics["pass_weak_RP"] += not compare_kw_lists(
                            keywords, gopnik_predictions["rep_change"]
                        )
                        gopnik_ep_metrics["pass_weak_FB"] += not compare_kw_lists(
                            keywords, gopnik_predictions["false_belief"]
                        )

            # 3.1 Interceptor (Eve) makes a guess
            interceptor_prompt = get_interceptor_prompt(
                info, hints, mode=interceptor_role_mode
            )

            if info["turn"] == 0 and interceptor_client.pass_system_as_user:
                interceptor_prompt = (
                    interceptor_client.system_prompt + interceptor_prompt
                )

            combined_history.append(
                {"role": "user", "content": f"[INTERCEPTOR] {interceptor_prompt}"}
            )

            guess_eve, interceptor_output, interceptor_info = (
                interceptor_client.chat_completion(
                    interceptor_prompt, None, None, hints, info["hint history"]
                )
            )
            interceptor_attempts.append(interceptor_info["attempts"])
            total_fails["interceptor"] += interceptor_info["fail"]
            if guess_eve is None:
                print("Error: Unable to get valid interceptor output")
                continue

            combined_history.append(
                {"role": "assistant", "content": f"[INTERCEPTOR] {interceptor_output}"}
            )

            # 4. Update environment
            new_code, rewards, done, info = env.step(hints, guess_bob, guess_eve)

            episode_turns += 1
            total_turns += 1

            turn_summary = f"""Turn {info["turn"]} summary:
    Code : {format_code(code)}
    Hints : {hints}
    Decoder guess : {format_code(guess_bob)}
    Interceptor guess : {format_code(guess_eve)}
    """

            # 5. Log turn information
            turn_intercept = False
            if info["miscommunications"] > prev_miscommunications:
                miscomm_turns.append(episode_turns)
                per_episode_miscomm_turns.append(episode_turns)
                prev_miscommunications = info["miscommunications"]
                turn_summary += "The Decoder failed to guess the code. The Encoder-Decoder team gets a Miscommunication token.\n"
            if info["interceptions"] > prev_interceptions:
                turn_intercept = True
                intercept_turns.append(episode_turns)
                per_episode_intercept_turns.append(episode_turns)
                prev_interceptions = info["interceptions"]
                turn_summary += "The Interceptor successfully guessed the code and gained an Interception token.\n"
                num_interceptions += 1

            if args.piaget and not isinstance(encoder_model, BaselineModel):
                if guess_eve == prediction_alice:
                    alice_predicted += 1
                    if turn_intercept:
                        alice_predicted_on_intercept += 1

                if env.code_to_string(prediction_alice) == code:
                    alice_predicted_code += 1

            info["prev_turn_summary"] = turn_summary

            if args.verbose:
                print(
                    f"""----------------------------------------------------
                Turn {info["turn"]}:
                code : {code}
                hints : {hints}
                guess_bob : {guess_bob}
                guess_eve : {guess_eve}
                Miscomm: {info["miscommunications"]}, Intercepts: {info["interceptions"]}
                done: {done}
                rewards: {rewards}
                ----------------------------------------------------
                """
                )

            code = new_code

            if done:
                if info["miscommunications"] == 2 and info["interceptions"] == 2:
                    end_cond["both"] += 1
                    both = True
                elif info["miscommunications"] == 2:
                    end_cond["miscomm"] += 1
                    miscommed = True
                elif info["interceptions"] == 2:
                    end_cond["intercept"] += 1
                    intercepted = True
                else:
                    end_cond["survived"] += 1
                    survived = True

                end_cond["turns"] = total_turns / (ep + 1)

                hint_hist = info["hint history"]
                game_over_message = f"""Game Over!
                The keywords and hints were:
                1. {keywords[0]} : {hint_hist[0]}
                2. {keywords[1]} : {hint_hist[1]}
                3. {keywords[2]} : {hint_hist[2]}
                4. {keywords[3]} : {hint_hist[3]}
                """
                combined_history.append(
                    {"role": "system", "content": game_over_message}
                )
                break

        changed_hints.append(changed_hint)

        alice_prediction_success.append(alice_predicted / episode_turns)
        if num_interceptions > 0:
            alice_prediction_success_on_intercept.append(
                alice_predicted_on_intercept / num_interceptions
            )
        alice_prediction_is_code.append(alice_predicted_code / episode_turns)

        if args.gopnik and gopnik_ep_metrics["num_failed_kw_pred"] > 0:
            gopnik_tom["weak_rep_change"].append(
                gopnik_ep_metrics["pass_weak_RP"]
                / gopnik_ep_metrics["num_failed_kw_pred"]
            )
            gopnik_tom["weak_false_belief"].append(
                gopnik_ep_metrics["pass_weak_FB"]
                / gopnik_ep_metrics["num_failed_kw_pred"]
            )
            gopnik_tom["self_other_consistency"].append(
                gopnik_ep_metrics["pass_SO_consistency"]
                / gopnik_ep_metrics["num_failed_kw_pred"]
            )
            gopnik_tom["strong_rep_change"].append(
                gopnik_ep_metrics["pass_strong_RP"]
                / gopnik_ep_metrics["num_failed_kw_pred"]
            )
            gopnik_tom["strong_false_belief"].append(
                gopnik_ep_metrics["pass_strong_FB"]
                / gopnik_ep_metrics["num_failed_kw_pred"]
            )

        all_histories.append(
            {
                "episode": ep,
                "episode_seed": ep_seed,
                "encoder_history": encoder_client.history,
                "decoder_history": decoder_client.history,
                "interceptor_history": interceptor_client.history,
                "combined_history": combined_history,
                "turns": episode_turns,
                "intercept_turns": per_episode_intercept_turns,
                "miscomm_turns": per_episode_miscomm_turns,
                "miscommed": miscommed,
                "intercepted": intercepted,
                "both": both,
                "survived": survived,
            }
        )

    avg_turns_per_episode = total_turns / args.num_episodes
    avg_intercept_turn = np.mean(intercept_turns) if intercept_turns else 0
    avg_miscomm_turn = np.mean(miscomm_turns) if miscomm_turns else 0
    avg_changed_hint = np.mean(changed_hints) / avg_turns_per_episode
    avg_predicted_success = (
        np.mean(alice_prediction_success)
        if len(alice_prediction_success) > 0
        else np.nan
    )
    avg_predicted_success_on_intercept = (
        np.mean(alice_prediction_success_on_intercept)
        if len(alice_prediction_success_on_intercept) > 0
        else np.nan
    )
    avg_prediction_is_code = (
        np.mean(alice_prediction_is_code)
        if len(alice_prediction_is_code) > 0
        else np.nan
    )
    avg_encoder_attempts = np.mean(encoder_attempts)
    avg_decoder_attempts = np.mean(decoder_attempts)
    avg_interceptor_attempts = np.mean(interceptor_attempts)

    end_cond["avg_changed_hint"] = avg_changed_hint
    end_cond["avg_predicted_success"] = avg_predicted_success
    end_cond["avg_predicted_success_on_intercept"] = avg_predicted_success_on_intercept
    end_cond["avg_prediction_is_code"] = avg_prediction_is_code
    end_cond["avg_turns_per_episode"] = avg_turns_per_episode
    end_cond["avg_encoder_attempts"] = avg_encoder_attempts
    end_cond["avg_decoder_attempts"] = avg_decoder_attempts
    end_cond["avg_interceptor_attempts"] = avg_interceptor_attempts
    end_cond["total_encoder_fails"] = total_fails["encoder"]
    end_cond["total_decoder_fails"] = total_fails["decoder"]
    end_cond["total_interceptor_fails"] = total_fails["interceptor"]

    if args.gopnik and not isinstance(interceptor_model, BaselineModel):
        for key, ep_pass_rates in gopnik_tom.items():
            end_cond["gopnik_" + key] = (
                np.mean(ep_pass_rates) if len(ep_pass_rates) > 0 else np.nan
            )

    return_dict = {
        "env_seed": env_seed,
        "model_seed": args.model_seed,
        "encoder_model": encoder_model.model_key,
        "decoder_model": decoder_model.model_key,
        "interceptor_model": interceptor_model.model_key,
        "encoder_mode": encoder_role_mode,
        "encoder_tom_mode": encoder_tom_mode,
        "decoder_mode": decoder_role_mode,
        "interceptor_mode": interceptor_role_mode,
        "encoder_system_mode": encoder_system_mode,
        "decoder_system_mode": decoder_system_mode,
        "interceptor_system_mode": interceptor_system_mode,
        "stats": end_cond,
        "histories": all_histories,
        "avg_turns_per_episode": avg_turns_per_episode,
        "avg_intercept_turn": avg_intercept_turn,
        "avg_miscomm_turn": avg_miscomm_turn,
        "avg_changed_hint": avg_changed_hint,
        "avg_predicted_success": avg_predicted_success,
        "avg_predicted_success_on_intercept": avg_predicted_success_on_intercept,
        "avg_prediction_is_code": avg_prediction_is_code,
        "avg_encoder_attempts": avg_encoder_attempts,
        "avg_decoder_attempts": avg_decoder_attempts,
        "avg_interceptor_attempts": avg_interceptor_attempts,
        "total_encoder_fails": total_fails["encoder"],
        "total_decoder_fails": total_fails["decoder"],
        "total_interceptor_fails": total_fails["interceptor"],
    }

    if args.gopnik and not isinstance(interceptor_model, BaselineModel):
        for key, ep_pass_rates in gopnik_tom.items():
            return_dict["gopnik_" + key] = (
                np.mean(ep_pass_rates) if len(ep_pass_rates) > 0 else np.nan
            )

    return return_dict


def replay_games(
    game_history,
    encoder_policy,
    decoder_policy,
    interceptor_policy,
    args=None,
):
    """
    Replay a Decrypto game with the specified policies.
    :param game_history: History of the game to replay
    :param encoder_policy: Tuple of (model, system_mode, role_mode, tom_mode)
    :param decoder_policy: Tuple of (model, system_mode, role_mode)
    :param interceptor_policy: Tuple of (model, system_mode, role_mode)
    :param args: Configuration arguments
    :return: Dictionary containing game replay statistics and histories
    """

    encoder_model, encoder_system_mode, encoder_role_mode, encoder_tom_mode = (
        encoder_policy
    )
    decoder_model, decoder_system_mode, decoder_role_mode = decoder_policy
    interceptor_model, interceptor_system_mode, interceptor_role_mode = (
        interceptor_policy
    )

    # Initialize clients for the roles that are not 'human'
    encoder_client = (
        RoleClient(
            "encoder",
            encoder_model,
            mode=encoder_system_mode,
            role_mode=encoder_role_mode,
            tom_mode=encoder_tom_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(encoder_model, Human)
        else None
    )

    decoder_client = (
        RoleClient(
            "decoder",
            decoder_model,
            mode=decoder_system_mode,
            role_mode=decoder_role_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(decoder_model, Human)
        else None
    )

    interceptor_client = (
        RoleClient(
            "interceptor",
            interceptor_model,
            mode=interceptor_system_mode,
            role_mode=interceptor_role_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(interceptor_model, Human)
        else None
    )

    summarized_history = game_history["summarized_history"]
    keywords = summarized_history[0]["keywords"]
    env = Decrypto()
    _, _, info = env.reset(seed=1789, keywords=keywords)
    # code = None  # We'll set this later
    # keywords = None  # We'll set this later

    # Initialize variables to keep track of game state
    end_cond = {"intercept": 0, "miscomm": 0, "both": 0, "survived": 0}
    total_turns = 0
    prev_interceptions = 0
    prev_miscommunications = 0
    intercept_turns = []
    miscomm_turns = []
    ending_turns = []
    code_history = []
    combined_history = []
    total_episodes = 1  # Since we're replaying one game

    for turn_data in summarized_history:
        turn = turn_data["turn"]
        print("Turn:", turn)
        code = turn_data["code"]
        hints = turn_data["hints"]

        code_numpy = np.array([int(num) for num in code])
        env.current_code = code_numpy
        code_history.append(code_numpy)
        env.code_history = code_history
        info["code_history"] = code_history[:-1]

        # Encoder provides hints
        encoder_prompt = get_encoder_prompt(
            info, keywords, code, mode=encoder_policy[1]
        )
        if encoder_client:
            # Generate new hints using the encoder_client
            if info["turn"] == 0 and encoder_client.pass_system_as_user:
                # DeepSeek takes the system prompt as part of user input instead.
                encoder_prompt = encoder_client.system_prompt + encoder_prompt

            hints, encoder_output, encoder_info = encoder_client.chat_completion(
                encoder_prompt, code, keywords, None, None
            )
            combined_history.append(
                {"role": "user", "content": f"[ENCODER] {encoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[ENCODER] {encoder_output}"}
            )
        else:
            print("Hints Alice: ", hints)
            # Use hints from the human game
            # combined_history.append({"role": "user", "content": f"[ENCODER] Provided hints"})
            combined_history.append(
                {"role": "user", "content": f"[ENCODER] {encoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[ENCODER] {hints}"}
            )

        # Decoder makes a guess
        decoder_prompt = get_decoder_prompt(
            keywords, info, hints, mode=decoder_policy[1]
        )
        if decoder_client:
            if info["turn"] == 0 and decoder_client.pass_system_as_user:
                decoder_prompt = decoder_client.system_prompt + decoder_prompt

            guess_bob, decoder_output, decoder_info = decoder_client.chat_completion(
                decoder_prompt, None, keywords, hints, info["hint history"]
            )
            combined_history.append(
                {"role": "user", "content": f"[DECODER] {decoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[DECODER] {decoder_output}"}
            )
        else:
            # Use guess from the human game
            guess_bob = turn_data["guess_bob"]
            # print('Guess Bob: ', guess_bob)
            # combined_history.append({"role": "user", "content": f"[DECODER] Made a guess"})
            combined_history.append(
                {"role": "user", "content": f"[DECODER] {decoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[DECODER] {guess_bob}"}
            )

        # Interceptor makes a guess
        interceptor_prompt = get_interceptor_prompt(
            info, hints, mode=interceptor_policy[1]
        )
        if interceptor_client:
            if info["turn"] == 0 and interceptor_client.pass_system_as_user:
                interceptor_prompt = (
                    interceptor_client.system_prompt + interceptor_prompt
                )

            # print("Interceptor Prompt:", interceptor_prompt)
            guess_eve, interceptor_output, interceptor_info = (
                interceptor_client.chat_completion(
                    interceptor_prompt, None, None, hints, info["hint history"]
                )
            )
            combined_history.append(
                {"role": "user", "content": f"[INTERCEPTOR] {interceptor_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[INTERCEPTOR] {interceptor_output}"}
            )
            print("Guess Eve:", guess_eve)
        else:
            # Use guess from the human game
            guess_eve = turn_data["guess_eve"]
            # combined_history.append({"role": "user", "content": f"[INTERCEPTOR] Made a guess"})
            combined_history.append(
                {"role": "user", "content": f"[INTERCEPTOR] {interceptor_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[INTERCEPTOR] {guess_eve}"}
            )

        # Now, simulate the environment's step
        new_code, rewards, done, info = env.step(hints, guess_bob, guess_eve)

        # Update tracking variables
        total_turns += 1

        turn_summary = f"""Turn {info["turn"]} summary:
    Code : {format_code(code)}
    Hints : {hints}
    Decoder guess : {format_code(guess_bob)}
    Interceptor guess : {format_code(guess_eve)}
    """

        if info["miscommunications"] > prev_miscommunications:
            miscomm_turns.append(total_turns)
            prev_miscommunications = info["miscommunications"]
            turn_summary += "The Decoder failed to guess the code. The Encoder-Decoder team gets a Miscommunication token.\n"
        if info["interceptions"] > prev_interceptions:
            intercept_turns.append(total_turns)
            prev_interceptions = info["interceptions"]
            turn_summary += "The Interceptor successfully guessed the code and gained an Interception token.\n"

        info["prev_turn_summary"] = turn_summary

        if done:
            # Determine how the game ended
            if info["miscommunications"] == 2 and info["interceptions"] == 2:
                end_cond["both"] += 1
            elif info["miscommunications"] == 2:
                end_cond["miscomm"] += 1
            elif info["interceptions"] == 2:
                end_cond["intercept"] += 1
            else:
                end_cond["survived"] += 1
            ending_turns.append(total_turns)
            hint_hist = info["hint history"]
            game_over_message = f"""Game Over!
            The keywords and hints were:
            1. {keywords[0]} : {hint_hist[0]}
            2. {keywords[1]} : {hint_hist[1]}
            3. {keywords[2]} : {hint_hist[2]}
            4. {keywords[3]} : {hint_hist[3]}
            """
            combined_history.append({"role": "system", "content": game_over_message})
            break

    # Aggregate statistics
    aggregated_stats = {
        "total_miscommunications": info.get("miscommunications", 0),
        "total_interceptions": info.get("interceptions", 0),
        "total_episodes": total_episodes,
        "average_turns_per_episode": total_turns / total_episodes
        if total_episodes > 0
        else 0,
        "average_intercept_turn": np.mean(intercept_turns) if intercept_turns else 0,
        "average_miscomm_turn": np.mean(miscomm_turns) if miscomm_turns else 0,
        "average_ending_turn": np.mean(ending_turns) if ending_turns else 0,
        "end_conditions": end_cond,
    }

    # Return results
    return {
        "encoder_model": encoder_policy[0],
        "decoder_model": decoder_policy[0],
        "interceptor_model": interceptor_policy[0],
        "aggregated_stats": aggregated_stats,
        "combined_history": combined_history,
    }


def run_games_with_humans(
    encoder_policy,
    decoder_policy,
    interceptor_policy,
    args=None,
):
    """
    Play Decrypto games with humans in the loop. Unlike the regular game loop, does not terminate after 2 intercepts.
    Used for collecting human data or playing with LLMs. For multiple human players, requires hot seat or similar setup.
    :param encoder_policy: Tuple of (model, system_mode, role_mode, tom_mode)
    :param decoder_policy: Tuple of (model, system_mode, role_mode)
    :param interceptor_policy: Tuple of (model, system_mode, role_mode)
    :param args: Configuration arguments
    :return: Dictionary containing game statistics and histories
    """

    encoder_model, encoder_system_mode, encoder_role_mode, encoder_tom_mode = (
        encoder_policy
    )
    decoder_model, decoder_system_mode, decoder_role_mode = decoder_policy
    interceptor_model, interceptor_system_mode, interceptor_role_mode = (
        interceptor_policy
    )
    all_histories = []

    encoder_client = (
        RoleClient(
            "encoder",
            encoder_model,
            mode=encoder_system_mode,
            role_mode=encoder_role_mode,
            tom_mode=encoder_tom_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(encoder_model, Human)
        else None
    )

    decoder_client = (
        RoleClient(
            "decoder",
            decoder_model,
            mode=decoder_system_mode,
            role_mode=decoder_role_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(decoder_model, Human)
        else None
    )

    interceptor_client = (
        RoleClient(
            "interceptor",
            interceptor_model,
            mode=interceptor_system_mode,
            role_mode=interceptor_role_mode,
            baseline_data_dir=args.baseline_data_dir,
            no_error_history=args.no_error_history,
        )
        if not isinstance(interceptor_model, Human)
        else None
    )

    end_cond = {"intercept": 0, "miscomm": 0, "both": 0, "survived": 0}
    total_turns = 0
    intercept_turns = []
    miscomm_turns = []
    ending_turns = []

    def get_human_input(
        prompt,
        parse_function=None,
        error_message="Invalid input. Please try again.",
        confirm_message="Are you sure you want to proceed with this input? (y/n): ",
        just_wait=False,
    ):
        while True:
            print(prompt)
            if just_wait:
                input("Press Enter to continue.")
                return None
            else:
                user_input = input("Enter your input as 'x, y, z' (without quotes): ")
                if parse_function:
                    try:
                        parsed_input = parse_function(user_input)
                        # Show the parsed input back to the user for confirmation
                        print(f"\nYou entered: {parsed_input}\n")
                        confirm = input(confirm_message).lower()
                        if confirm == "y":
                            return (
                                parsed_input,
                                user_input,
                            )  # Return both parsed and original input
                        else:
                            os.system("clear")
                            print("Let's try again.\n")
                            continue  # Continue the loop to reprompt
                    except ValueError as ve:
                        os.system("clear")
                        print(f"{error_message} {ve}\n")
                else:
                    return (
                        user_input,
                        user_input,
                    )  # Return user_input in both positions for consistency

    def parse_human_hints(input_str):
        hints = [h.strip() for h in input_str.split(",")]
        if not hints or any(not h for h in hints) or len(hints) != 3:
            raise ValueError("You have to provide exactly three hints.")
        return hints

    def parse_human_guess(input_str):
        try:
            guess = [int(x.strip()) for x in input_str.split(",")]
        except ValueError:
            raise ValueError("Input must be numbers.")
        if len(guess) != 3:
            raise ValueError("You must provide exactly three numbers.")
        return guess

    for ep in range(args.num_episodes):
        env = Decrypto()
        ep_seed = ep + args.env_seed * args.num_episodes
        code, keywords, info = env.reset(seed=ep_seed)

        print("\n\n NEW GAME \n\n")

        if encoder_client:
            encoder_client.reset_used_hints()

        combined_history = []
        summarized_history = []

        if encoder_client:
            encoder_client.reset_history()
        if decoder_client:
            decoder_client.reset_history()
        if interceptor_client:
            interceptor_client.reset_history()

        episode_turns = 0
        prev_interceptions = 0
        prev_miscommunications = 0
        episode_over = False  # Flag to check if the game-over condition has been logged
        ending_turn = None
        intercepted = False
        miscommed = False
        survived = False
        both = False
        per_episode_intercept_turns = []
        per_episode_miscomm_turns = []

        for step in range(8):
            # Encoder provides hints
            encoder_prompt = get_encoder_prompt(
                info, keywords, code, mode=encoder_role_mode
            )

            if isinstance(encoder_model, Human):
                if step == 0:
                    encoder_sys_prompt = get_system_prompt("encoder", encoder_role_mode)
                    encoder_prompt = (
                        encoder_sys_prompt + "\n ------ \n\n" + encoder_prompt
                    )
                hints, encoder_output = get_human_input(
                    encoder_prompt,
                    parse_function=parse_human_hints,
                    error_message="Invalid format. Please enter hints separated by commas (e.g., 'hint1, hint2, hint3').",
                    confirm_message="Are you sure you want to provide these hints? (y/n): ",
                )
            else:
                if info["turn"] == 0 and encoder_client.pass_system_as_user:
                    encoder_prompt = encoder_client.system_prompt + encoder_prompt
                hints, encoder_output, encoder_info = encoder_client.chat_completion(
                    encoder_prompt, code, keywords, None, None
                )

            combined_history.append(
                {"role": "user", "content": f"[ENCODER] {encoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[ENCODER] {encoder_output}"}
            )
            os.system("clear")

            # Decoder makes a guess
            decoder_prompt = get_decoder_prompt(
                keywords, info, hints, mode=decoder_role_mode
            )
            if isinstance(decoder_model, Human):
                if step == 0:
                    decoder_sys_prompt = get_system_prompt("decoder", decoder_role_mode)
                    decoder_prompt = (
                        decoder_sys_prompt + "\n ------ \n\n" + decoder_prompt
                    )
                guess_bob, decoder_output = get_human_input(
                    decoder_prompt,
                    parse_function=parse_human_guess,
                    error_message="Invalid format. Please enter three numbers between 1 and 4, separated by commas (e.g., '1, 2, 3').",
                    confirm_message="Are you sure you want to provide this guess? (y/n): ",
                )
            else:
                if info["turn"] == 0 and decoder_client.pass_system_as_user:
                    decoder_prompt = decoder_client.system_prompt + decoder_prompt

                guess_bob, decoder_output, decoder_info = (
                    decoder_client.chat_completion(
                        decoder_prompt, None, keywords, hints, info["hint history"]
                    )
                )

            combined_history.append(
                {"role": "user", "content": f"[DECODER] {decoder_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[DECODER] {decoder_output}"}
            )
            os.system("clear")

            # Interceptor makes a guess
            interceptor_prompt = get_interceptor_prompt(
                info, hints, mode=interceptor_role_mode
            )

            if isinstance(interceptor_model, Human):
                if step == 0:
                    interceptor_sys_prompt = get_system_prompt(
                        "interceptor", interceptor_role_mode
                    )
                    interceptor_prompt = (
                        interceptor_sys_prompt + "\n ------ \n\n" + interceptor_prompt
                    )
                guess_eve, interceptor_output = get_human_input(
                    interceptor_prompt,
                    parse_function=parse_human_guess,
                    error_message="Invalid format. Please enter three numbers between 1 and 4, separated by commas (e.g., '1, 2, 3').",
                    confirm_message="Are you sure you want to provide this guess? (y/n): ",
                )
            else:
                if info["turn"] == 0 and interceptor_client.pass_system_as_user:
                    interceptor_prompt = (
                        interceptor_client.system_prompt + interceptor_prompt
                    )

                guess_eve, interceptor_output, interceptor_info = (
                    interceptor_client.chat_completion(
                        interceptor_prompt, None, None, hints, info["hint history"]
                    )
                )

            combined_history.append(
                {"role": "user", "content": f"[INTERCEPTOR] {interceptor_prompt}"}
            )
            combined_history.append(
                {"role": "assistant", "content": f"[INTERCEPTOR] {interceptor_output}"}
            )
            os.system("clear")

            summarized_history.append(
                {
                    "keywords": keywords,
                    "turn": episode_turns,
                    "code": code,
                    "hints": hints,
                    "guess_bob": guess_bob,
                    "guess_eve": guess_eve,
                }
            )

            new_code, rewards, done, info = env.step(hints, guess_bob, guess_eve)

            episode_turns += 1
            total_turns += 1

            turn_summary = f"""Turn {info["turn"]} summary:
    Code : {format_code(code)}
    Hints : {hints}
    Decoder guess : {format_code(guess_bob)}
    Interceptor guess : {format_code(guess_eve)}
    """

            if info["miscommunications"] > prev_miscommunications:
                miscomm_turns.append(episode_turns)
                per_episode_miscomm_turns.append(episode_turns)
                prev_miscommunications = info["miscommunications"]
                turn_summary += "The Decoder failed to guess the code. The Encoder-Decoder team gets a Miscommunication token.\n"
            if info["interceptions"] > prev_interceptions:
                intercept_turns.append(episode_turns)
                per_episode_intercept_turns.append(episode_turns)
                prev_interceptions = info["interceptions"]
                turn_summary += "The Interceptor successfully guessed the code and gained an Interception token.\n"

            info["prev_turn_summary"] = turn_summary

            os.system("clear")

            get_human_input(turn_summary, just_wait=True)
            os.system("clear")

            code = new_code

            if done and not episode_over:
                episode_over = True  # Set the flag to True to avoid multiple logs
                ending_turn = episode_turns

                if info["miscommunications"] == 2 and info["interceptions"] == 2:
                    end_cond["both"] += 1
                    both = True
                    break
                elif info["miscommunications"] == 2:
                    end_cond["miscomm"] += 1
                    miscommed = True
                    break
                elif info["interceptions"] == 2:
                    end_cond["intercept"] += 1
                    intercepted = True
                else:
                    end_cond["survived"] += 1
                    survived = True

        hint_hist = info["hint history"]
        game_over_message = f"""Game Over!
        The keywords and hints were:
        1. {keywords[0]} : {hint_hist[0]}
        2. {keywords[1]} : {hint_hist[1]}
        3. {keywords[2]} : {hint_hist[2]}
        4. {keywords[3]} : {hint_hist[3]}
        """
        combined_history.append({"role": "system", "content": game_over_message})

        print(game_over_message)

        ending_turns.append(ending_turn)

        all_histories.append(
            {
                "episode": ep,
                "combined_history": combined_history,
                "summarized_history": summarized_history,
                "turns": episode_turns,
                "ending_turn": ending_turn,
                "intercept_turns": intercept_turns,
                "miscomm_turns": miscomm_turns,
                "miscommed": miscommed,
                "intercepted": intercepted,
                "both": both,
                "survived": survived,
            }
        )

    avg_turns_per_episode = total_turns / args.num_episodes
    avg_intercept_turn = np.mean(intercept_turns) if intercept_turns else 0
    avg_miscomm_turn = np.mean(miscomm_turns) if miscomm_turns else 0
    avg_ending_turn = (
        np.mean([et for et in ending_turns if et is not None]) if ending_turns else 0
    )
    return {
        "encoder_model": encoder_policy[0],
        "decoder_model": decoder_policy[0],
        "interceptor_model": interceptor_policy[0],
        "encoder_mode": encoder_role_mode,
        "decoder_mode": decoder_role_mode,
        "interceptor_mode": interceptor_role_mode,
        "stats": end_cond,
        "histories": all_histories,
        "avg_turns_per_episode": avg_turns_per_episode,
        "avg_intercept_turn": avg_intercept_turn,
        "avg_miscomm_turn": avg_miscomm_turn,
        "avg_ending_turn": avg_ending_turn,
    }


def save_experiment_results(results, args, base_path="results/"):
    """
    Save the results of the experiment to a JSON file.
    :param results: Dictionary containing the results of the experiment
    :param args: Configuration arguments
    :param base_path: Base path for saving the results
    """
    experiment_name = f"model_seed{results['model_seed']}/"
    experiment_name += f"{results['encoder_model']}_{results['decoder_model']}_{results['interceptor_model']}"
    experiment_name += f"_{results['encoder_mode']}{results['decoder_mode']}{results['interceptor_mode']}"
    experiment_name += f"_{results['encoder_system_mode']}{results['decoder_system_mode']}{results['interceptor_system_mode']}"
    if args.theory_of_mind and "encoder_tom_mode" in results:
        experiment_name += f"_ToM{results['encoder_tom_mode']}"
    if "env_seed" in results:
        experiment_name += f"/env_seed{results['env_seed']}"
    experiment_path = os.path.join(base_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Save statistics
    with open(os.path.join(experiment_path, "stats.json"), "w") as f:
        json.dump(results["stats"], f, indent=2)

    # Save histories
    for episode in results["histories"]:
        episode_path = os.path.join(experiment_path, f"episode_{episode['episode']}")
        os.makedirs(episode_path, exist_ok=True)

        for role in ["encoder", "decoder", "interceptor"]:
            with open(os.path.join(episode_path, f"{role}_history.json"), "w") as f:
                json.dump(episode[f"{role}_history"], f, indent=2)
        with open(os.path.join(episode_path, "combined_history.json"), "w") as f:
            json.dump(episode["combined_history"], f, indent=2)

        # Save per-episode statistics
        episode_stats = {
            "turns": episode["turns"],
            "intercept_turns": episode["intercept_turns"],
            "miscomm_turns": episode["miscomm_turns"],
            "miscommed": episode["miscommed"],
            "intercepted": episode["intercepted"],
            "both": episode["both"],
            "survived": episode["survived"],
        }
        with open(
            os.path.join(episode_path, f"stats_episode_{episode['episode']}.json"), "w"
        ) as f:
            json.dump(episode_stats, f, indent=2)


def run_experiments(cfg):
    """
    Orchestrate, monitor and log Decrypto games with the given configuration, sweeping over model matchups in parallel.
    :param cfg: Configuration object containing the experiment parameters
    """
    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.exp_name is None:
        base_path = f"results/experiment_{timestamp}"
    else:
        base_path = f"results/{cfg.exp_name}"
    os.makedirs(base_path, exist_ok=True)

    if cfg.get_models_from_slurm:
        local_models_info = get_available_servers()

        for model_info in local_models_info:
            local_model = LocalModel(
                model_key=model_info["model_key"],
                model_id=model_info["model_id"],
                urls=model_info["urls"],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                model_seed=cfg.model_seed,
                job_ids=model_info["job_ids"],
            )
            cfg.models.append(local_model)
    models = cfg.models

    # TODO: Make configs for this
    # if cfg.include_claude_haiku:
    #     models.update({"claude3_haiku": "claude-3-haiku-20240307"})
    # if cfg.include_claude_sonnet:
    #     models.update({"claude3.5_sonnet": "claude-3-5-sonnet-20240620"})
    # if cfg.include_gpt_4:
    #     models.update({"gpt-4o": "gpt-4o"})

    # Download zip files on single process before we load on multiprocess
    for model in models:
        if isinstance(model, BaselineModel):
            if model.model_id.lower() == "glove":
                load_model_and_tokenizer("glove", cfg.baseline_data_dir)
            elif model.model_id.lower() == "word2vec":
                load_model_and_tokenizer("word2vec", cfg.baseline_data_dir)

    # Policy is determined by LLM and prompt
    system_prompt_modes = cfg.system_prompt_modes
    role_prompt_modes = cfg.role_prompt_modes
    tom_modes = cfg.tom_modes
    if not cfg.theory_of_mind:
        # No ToM modes in this case
        tom_modes = [-1]

    # Prepare encoder policies
    encoder_policies = []
    for model in models:
        if isinstance(model, BaselineModel):
            # Baseline models have fixed modes
            encoder_policies.append((model, 0, 0, 0))
        else:
            encoder_policies.extend(
                [
                    (model, sys_mode, role_mode, tom_mode)
                    for sys_mode in system_prompt_modes
                    for role_mode in role_prompt_modes
                    for tom_mode in tom_modes
                ]
            )

    # Prepare decoder policies
    decoder_policies = []
    for model in models:
        if isinstance(model, BaselineModel):
            decoder_policies.append((model, 0, 0))
        else:
            decoder_policies.extend(
                [
                    (model, sys_mode, role_mode)
                    for sys_mode in system_prompt_modes
                    for role_mode in role_prompt_modes
                ]
            )

    # Prepare interceptor policies with modes set to 0
    interceptor_policies = []
    for model in models:
        interceptor_policies.append((model, 0, 0))

    # Generate combinations (num_combos = n_enc x n_dec x n_int x n_env_seeds)

    if len(cfg.fixed_encoder) > 0:
        # filter encoder policies to only include the fixed encoder
        fixed_encoder_model = next(
            (m for m in models if m.model_key == cfg.fixed_encoder), None
        )
        assert fixed_encoder_model is not None, (
            f"Fixed encoder '{cfg.fixed_encoder}' not found in models: {[m.model_key for m in models]}."
        )
        encoder_policies = [
            ep for ep in encoder_policies if ep[0] == fixed_encoder_model
        ]  # overwrite encoder_policies to only include the fixed encoder
    if len(cfg.fixed_decoder) > 0:
        # filter decoder policies to only include the fixed decoder
        fixed_decoder_model = next(
            (m for m in models if m.model_key == cfg.fixed_decoder), None
        )
        assert fixed_decoder_model is not None, (
            f"Fixed decoder '{cfg.fixed_decoder}' not found in models: {[m.model_key for m in models]}."
        )
        decoder_policies = [
            dp for dp in decoder_policies if dp[0] == fixed_decoder_model
        ]  # overwrite decoder_policies to only include the fixed decoder
    if len(cfg.fixed_interceptor) > 0:
        # filter interceptor policies to only include the fixed interceptor
        fixed_interceptor_model = next(
            (m for m in models if m.model_key == cfg.fixed_interceptor), None
        )
        assert fixed_interceptor_model is not None, (
            f"Fixed interceptor '{cfg.fixed_interceptor}' not found in models: {[m.model_key for m in models]}."
        )
        interceptor_policies = [
            ip for ip in interceptor_policies if ip[0] == fixed_interceptor_model
        ]  # overwrite interceptor_policies to only include the fixed interceptor

    combinations = list(
        itertools.product(
            encoder_policies, decoder_policies, interceptor_policies, cfg.env_seed
        )
    )

    if cfg.match_encoder_decoder:
        # Only keep combos where encoder == decoder
        combinations = [
            c
            for c in combinations
            if (c[0][0] == c[1][0] and c[0][1] == c[1][1] and c[0][2] == c[1][2])
        ]
    elif cfg.match_encoder_decoder_models_only:
        # Only keep combos where encoder_model == decoder_model
        combinations = [c for c in combinations if (c[0][0] == c[1][0])]
    if len(cfg.filter_model) > 0:
        # Only keep combinations that contain the filter model
        filter_model_instance = next(
            (m for m in models if m.model_key == cfg.filter_model), None
        )
        assert filter_model_instance is not None, "Filter model not found in models."
        combinations = [
            c
            for c in combinations
            if (
                c[0][0] == filter_model_instance
                or c[1][0] == filter_model_instance
                or c[2][0] == filter_model_instance
            )
        ]

    print(f"Starting {len(combinations)} combinations:")
    for i, combo in enumerate(combinations):
        print(i, combo_to_string(combo))
    np.random.shuffle(combinations)

    results = []
    total_experiments = len(combinations)
    completed_experiments = 0

    print(
        f"Starting {total_experiments} experiments with {cfg.num_episodes} episodes each..."
    )
    print(f"Results will be saved in folder: {base_path}")

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=total_experiments) as executor:
        future_to_combo = {
            executor.submit(run_games, *combo, cfg): combo for combo in combinations
        }
        for future in tqdm(
            as_completed(future_to_combo),
            total=len(combinations),
            desc="Overall Progress",
        ):
            combo = future_to_combo[future]
            combo_string = combo_to_string(combo)
            try:
                result = future.result()
                results.append(result)
                save_experiment_results(result, cfg, base_path)

                completed_experiments += 1
                elapsed_time = time.time() - start_time
                avg_time_per_experiment = elapsed_time / completed_experiments
                estimated_time_remaining = (
                    total_experiments - completed_experiments
                ) * avg_time_per_experiment

                print(f"\nCompleted experiment: {combo_string}")
                print(f"Progress: {completed_experiments}/{total_experiments}")
                print(
                    f"Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes"
                )
                print(
                    f"Results: Intercepts: {result['stats']['intercept']}, Miscomms: {result['stats']['miscomm']}, Both: {result['stats']['both']}, Survived: {result['stats']['survived']}"
                )
                print(f"Avg Turns per Episode: {result['avg_turns_per_episode']:.2f}")
                print(f"Avg Turn for Intercepts: {result['avg_intercept_turn']:.2f}")
                print(
                    f"Avg Turn for Miscommunications: {result['avg_miscomm_turn']:.2f}"
                )
                print(
                    f"Avg Ratio of Hint Changes to Avg Turn Length: {result['avg_changed_hint']:.2f}"
                )
            except Exception as exc:
                print(f"{combo_string} generated an exception: {exc}")
                traceback.print_exc()

    # Create a summary DataFrame
    summary_data = []
    for result in results:
        summary = {
            "env_seed": result["env_seed"],
            "encoder": result["encoder_model"],
            "encoder_mode": result["encoder_mode"],
            "encoder_system_mode": result["encoder_system_mode"],
            "decoder": result["decoder_model"],
            "decoder_mode": result["decoder_mode"],
            "decoder_system_mode": result["decoder_system_mode"],
            "interceptor": result["interceptor_model"],
            "interceptor_mode": result["interceptor_mode"],
            "interceptor_system_mode": result["interceptor_system_mode"],
            "intercepts": result["stats"]["intercept"],
            "miscomms": result["stats"]["miscomm"],
            "both": result["stats"]["both"],
            "survived": result["stats"]["survived"],
            "avg_turns_per_episode": result["avg_turns_per_episode"],
            "avg_intercept_turn": result["avg_intercept_turn"],
            "avg_miscomm_turn": result["avg_miscomm_turn"],
            "avg_changed_hint": result["avg_changed_hint"],
            "avg_encoder_attempts": result["avg_encoder_attempts"],
            "avg_decoder_attempts": result["avg_decoder_attempts"],
            "avg_interceptor_attempts": result["avg_interceptor_attempts"],
            "total_encoder_fails": result["total_encoder_fails"],
            "total_decoder_fails": result["total_decoder_fails"],
            "total_interceptor_fails": result["total_interceptor_fails"],
        }
        if cfg.theory_of_mind:
            summary["encoder_tom_mode"] = result["encoder_tom_mode"]
        if cfg.piaget:
            summary["avg_predicted_success"] = result["avg_predicted_success"]
            summary["avg_predicted_success_on_intercept"] = result[
                "avg_predicted_success_on_intercept"
            ]
            summary["avg_prediction_is_code"] = result["avg_prediction_is_code"]
        if cfg.gopnik:
            for key, value in result.items():
                if "gopnik" in key:
                    summary[key] = value

        summary_data.append(summary)

    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(base_path, "experiment_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"\nExperiments completed. Results saved in '{base_path}' folder.")
    print(f"Summary saved as '{summary_file}'.")
    print("Total Time: " + f"{time.time() - start_time}")


def run_replay_experiments(cfg):
    """
    Replay saved games, replacing one of the players by a model. Mainly used to evaluate LLMs on human data.
    :param cfg: Configuration object containing the experiment parameters
    """
    # Load all game histories from the specified folder
    game_files = glob.glob(os.path.join(cfg.replay_data_dir, "*.pkl"))
    if not game_files:
        print(f"No game files found in {cfg.replay_data_dir}")
        return

    models_to_eval = []
    if cfg.replay_as_decoder:
        models_to_eval.extend([(m, "decoder") for m in cfg.models])
    if cfg.replay_as_interceptor:
        models_to_eval.extend([(m, "interceptor") for m in cfg.models])

    for model, role in models_to_eval:
        # Human models are placeholders for replaying actions from game history
        if role == "decoder":
            models = [Human(model_key="history"), model, Human(model_key="history")]
        elif role == "interceptor":
            models = [Human(model_key="history"), Human(model_key="history"), model]

        # Initialize aggregated statistics
        total_aggregated_stats = {
            "total_miscommunications": 0,
            "total_interceptions": 0,
            "total_episodes": 0,
            "total_turns": 0,
            "intercept_turns": [],
            "miscomm_turns": [],
            "ending_turns": [],
            "end_cond": {"intercept": 0, "miscomm": 0, "both": 0, "survived": 0},
        }
        all_combined_histories = []

        for game_file in tqdm(game_files, desc="Processing game files"):
            # Load the game data
            with open(game_file, "rb") as f:
                game_data = pkl.load(f)
            if "histories" not in game_data:
                continue  # Skip files without 'histories'

            # Process each game history in the game data
            for game_history in game_data["histories"]:
                # Run the replay for each game
                results = replay_games(
                    game_history,
                    encoder_policy=(models[0], 0, 0, 0),
                    decoder_policy=(models[1], 0, 0),
                    interceptor_policy=(models[2], 0, 0),
                    args=cfg,
                )

                # Aggregate the statistics
                aggregated_stats = results["aggregated_stats"]
                total_aggregated_stats["total_miscommunications"] += aggregated_stats[
                    "total_miscommunications"
                ]
                total_aggregated_stats["total_interceptions"] += aggregated_stats[
                    "total_interceptions"
                ]
                total_aggregated_stats["total_episodes"] += aggregated_stats[
                    "total_episodes"
                ]
                total_aggregated_stats["total_turns"] += (
                    aggregated_stats["average_turns_per_episode"]
                    * aggregated_stats["total_episodes"]
                )
                if aggregated_stats["average_intercept_turn"] > 0:
                    total_aggregated_stats["intercept_turns"].append(
                        aggregated_stats["average_intercept_turn"]
                    )
                if aggregated_stats["average_miscomm_turn"] > 0:
                    total_aggregated_stats["miscomm_turns"].append(
                        aggregated_stats["average_miscomm_turn"]
                    )
                if aggregated_stats["average_ending_turn"] > 0:
                    total_aggregated_stats["ending_turns"].append(
                        aggregated_stats["average_ending_turn"]
                    )
                for key in total_aggregated_stats["end_cond"]:
                    total_aggregated_stats["end_cond"][key] += aggregated_stats[
                        "end_conditions"
                    ].get(key, 0)
                all_combined_histories.extend(results["combined_history"])

        # Compute averages
        if total_aggregated_stats["total_episodes"] > 0:
            total_aggregated_stats["average_turns_per_episode"] = (
                total_aggregated_stats["total_turns"]
                / total_aggregated_stats["total_episodes"]
            )
        else:
            total_aggregated_stats["average_turns_per_episode"] = 0
        total_aggregated_stats["average_intercept_turn"] = (
            np.mean(total_aggregated_stats["intercept_turns"])
            if total_aggregated_stats["intercept_turns"]
            else 0
        )
        total_aggregated_stats["average_miscomm_turn"] = (
            np.mean(total_aggregated_stats["miscomm_turns"])
            if total_aggregated_stats["miscomm_turns"]
            else 0
        )
        total_aggregated_stats["average_ending_turn"] = (
            np.mean(total_aggregated_stats["ending_turns"])
            if total_aggregated_stats["ending_turns"]
            else 0
        )

        # Save the results
        os.makedirs(f"results/human_replay/{cfg.exp_name}", exist_ok=True)
        output_file = f"results/human_replay/{cfg.exp_name}/{models[0].model_key}_{models[1].model_key}_{models[2].model_key}_replay.pkl"
        with open(output_file, "wb") as f:
            pkl.dump(
                {
                    "encoder_model": models[0].model_key,
                    "decoder_model": models[1].model_key,
                    "interceptor_model": models[2].model_key,
                    "aggregated_stats": total_aggregated_stats,
                    "combined_history": all_combined_histories,
                },
                f,
            )

        # Print aggregated statistics
        if cfg.verbose:
            print("\nAggregated Statistics:")
            print(f"Total Episodes: {total_aggregated_stats['total_episodes']}")
            print(
                f"Total Miscommunications: {total_aggregated_stats['total_miscommunications']}"
            )
            print(
                f"Total Interceptions: {total_aggregated_stats['total_interceptions']}"
            )
            print(
                f"Average Turns per Episode: {total_aggregated_stats['average_turns_per_episode']:.2f}"
            )
            print(
                f"Average Interception Turn: {total_aggregated_stats['average_intercept_turn']:.2f}"
            )
            print(
                f"Average Miscommunication Turn: {total_aggregated_stats['average_miscomm_turn']:.2f}"
            )
            print(
                f"Average Ending Turn: {total_aggregated_stats['average_ending_turn']:.2f}"
            )
            for condition, count in total_aggregated_stats["end_cond"].items():
                print(f"  {condition.capitalize()}: {count}")

    print(f"Done. Results saved to: results/human_replay/{cfg.exp_name}")
