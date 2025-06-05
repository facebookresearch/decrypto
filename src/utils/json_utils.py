# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re

def extract_json_answer(content):
    json_match = re.search(r"ANSWER:\s*(\{.*\})", content, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            # Check for encoder format (hints)
            if "hints" in json_data:
                hints = json_data["hints"]
                if (
                    isinstance(hints, list)
                    and len(hints) == 3
                    and all(isinstance(hint, str) for hint in hints)
                ):
                    return json_data
                else:
                    print("Invalid hints format. Expected exactly three string hints.")
            # Check for decoder/interceptor format (guess)
            elif "guess" in json_data:
                guess = json_data["guess"]
                if (
                    isinstance(guess, str)
                    and re.match(r"^[1-4]-[1-4]-[1-4]$", guess)
                    and len(guess.split("-")) == 3
                ):
                    return json_data
                else:
                    print(
                        "Invalid guess format. Expected 'X-Y-Z' where X, Y, Z are unique digits from 1 to 4."
                    )
            elif "keywords" in json_data:
                keywords = json_data["keywords"]
                if (
                    isinstance(keywords, list)
                    and len(keywords) == 4
                    and all(isinstance(kw, str) for kw in keywords)
                ):
                    return json_data
                else:
                    print(
                        "Invalid hints format. Expected exactly four string keywords."
                    )

            else:
                print(
                    "Invalid JSON structure. Missing 'hints', 'guess' or 'keywords' key."
                )
            return None
        except Exception:
            print("Failed to parse JSON.")
            return None
    else:
        print("No JSON answer found in the content.")
        return None


def compare_kw_lists(list1, list2):
    # replace "unknown" with empty string and remove leading/trailing whitespace
    list1 = [s.replace("unknown", "").strip() for s in list1]
    list2 = [s.replace("unknown", "").strip() for s in list2]
    # compare ignoring case and whitespace
    return all(s1.lower() == s2.lower() for s1, s2 in zip(list1, list2))
