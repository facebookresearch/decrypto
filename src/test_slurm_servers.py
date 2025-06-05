# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

from openai import OpenAI

from utils.server import get_available_servers

models = get_available_servers()
print()

for model in models:
    for i, url in enumerate(model["urls"]):
        try:
            client = OpenAI(
                api_key="dummy_key",
                base_url=url,
            )
            start = time.time()
            completion = client.chat.completions.create(
                model=model["model_id"],
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'Hello World!' in one line, but make it funky.",
                    }
                ],
                max_tokens=30,
            )
            print(
                f" -      {model['model_key']} ({model['model_id']}):",
                completion.choices[0].message.content,
                f"| Reply time: {time.time() - start:.2f} sec.",
            )
        except Exception as e:
            print(
                f"[!] {model['model_key']} ({model['model_id']}): Server not responsive. Error: {e}"
            )


print("\nAVAILABLE SERVERS")
for model in models:
    print(f" - {model['model_key']} ({model['model_id']}):")
    for url in model["urls"]:
        print(f"        - {url}")
    print()
