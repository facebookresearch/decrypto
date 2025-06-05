# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os


def generate_html(conversation_data, output_file):
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decrypto Game Conversation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .user {
            background-color: #e6f3ff;
            border-left: 5px solid #3498db;
        }
        .assistant {
            background-color: #f0f0f0;
            border-left: 5px solid #2ecc71;
        }
        .system {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .role {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .content {
            white-space: pre-wrap;
        }
        .encoder { color: #3498db; }
        .decoder { color: #2ecc71; }
        .interceptor { color: #e74c3c; }
    </style>
</head>
<body>
    <div id="chat-container"></div>

    <script>
        const conversation = {conversation_json};

        const chatContainer = document.getElementById('chat-container');

        conversation.forEach(message => {{
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', message.role);

            const roleDiv = document.createElement('div');
            roleDiv.classList.add('role');
            roleDiv.textContent = message.role.toUpperCase();

            const contentDiv = document.createElement('div');
            contentDiv.classList.add('content');
            
            // Highlight [ENCODER], [DECODER], and [INTERCEPTOR] tags
            const content = message.content.replace(/\\[(ENCODER|DECODER|INTERCEPTOR)\\]/g, (match, p1) => `<span class="${p1.toLowerCase()}">[${p1}]</span>`);
            contentDiv.innerHTML = content;

            messageDiv.appendChild(roleDiv);
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
        }});
    </script>
</body>
</html>
    """

    # Convert the conversation data to a JSON string
    conversation_json = json.dumps(conversation_data)

    # Replace the placeholder in the template with the actual JSON data
    html_content = html_template.replace("{conversation_json}", conversation_json)

    # Write the HTML content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)


def process_episode(
    experiment_dir,
    model_seed,
    model_combo,
    env_seed,
    episode,
    output_dir,
    output_name=None,
):
    episode_path = os.path.join(
        experiment_dir,
        f"model_seed{model_seed}",
        model_combo,
        f"env_seed{env_seed}",
        f"episode_{episode}",
    )
    combined_history_file = os.path.join(episode_path, "combined_history.json")

    if not os.path.exists(combined_history_file):
        print(
            f"Combined history file not found for env_seed {env_seed}, episode {episode} of {model_combo}"
        )
        print(combined_history_file)
        return

    with open(combined_history_file, "r", encoding="utf-8") as f:
        conversation_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    file_name = (
        output_name
        if output_name is not None
        else f"{model_combo}_episode_{episode}.html"
    )
    output_file = os.path.join(output_dir, file_name)
    generate_html(conversation_data, output_file)
    print(f"Generated HTML file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization for Decrypto game conversations"
    )
    parser.add_argument(
        "--experiment_dir", help="Path to the experiment results directory"
    )
    parser.add_argument(
        "--model_combo", help="Model combination (e.g., llama_llama_llama_000_000)"
    )
    parser.add_argument("--model_seed", type=int, default=0, help="Model seed")
    parser.add_argument("--env_seed", type=int, default=0, help="Env seed")
    parser.add_argument("--episode", type=int, default=0, help="Episode number")
    parser.add_argument(
        "--output_dir",
        default="../game_transcripts",
        help="Directory to save the generated HTML files",
    )
    parser.add_argument(
        "--output_name", default=None, help="Custom name for the transcript file"
    )

    args = parser.parse_args()

    process_episode(
        args.experiment_dir,
        args.model_seed,
        args.model_combo,
        args.env_seed,
        args.episode,
        args.output_dir,
        args.output_name,
    )


if __name__ == "__main__":
    main()
