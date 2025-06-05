# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl


def load_results(file_path):
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data


def calculate_rates(stats):
    total_episodes = stats["total_episodes"]
    total_interceptions = stats.get("total_interceptions", 0)
    total_miscommunications = stats.get("total_miscommunications", 0)

    interception_rate = (
        (total_interceptions / 2 / total_episodes) * 100 if total_episodes else 0
    )
    miscommunication_rate = (
        (total_miscommunications / 2 / total_episodes) * 100 if total_episodes else 0
    )

    return interception_rate, miscommunication_rate


def generate_latex_table_interceptor(stats, model_name):
    ir, _ = calculate_rates(stats)
    table = r"""
\begin{table}[h!]
\centering
\begin{tabular}{lccccc}
\hline
\textbf{Interceptor Model} & \textbf{Total Episodes} & \textbf{Total Interceptions} & \textbf{Interception Rate} & \textbf{Avg.\ Interception Turn} & \textbf{Avg.\ Ending Turn} \\\hline
%s & %d & %d & %.2f\\%% & %.2f & %.2f \\\hline
\end{tabular}
\caption{Performance of %s as the Interceptor in the Decrypto game.}
\label{table:interceptor_performance}
\end{table}
""" % (
        model_name,
        stats["total_episodes"],
        stats["total_interceptions"],
        ir,
        stats["average_intercept_turn"],
        stats["average_ending_turn"],
        model_name,
    )
    return table


def generate_latex_table_decoder(stats, model_name):
    _, mr = calculate_rates(stats)
    table = r"""
\begin{table}[h!]
\centering
\begin{tabular}{lccccc}
\hline
\textbf{Decoder Model} & \textbf{Total Episodes} & \textbf{Total Miscommunications} & \textbf{Miscommunication Rate} & \textbf{Avg.\ Miscomm.\ Turn} & \textbf{Avg.\ Ending Turn} \\\hline
%s & %d & %d & %.2f\\%% & %.2f & %.2f \\\hline
\end{tabular}
\caption{Performance of %s as the Decoder in the Decrypto game.}
\label{table:decoder_performance}
\end{table}
""" % (
        model_name,
        stats["total_episodes"],
        stats["total_miscommunications"],
        mr,
        stats["average_miscomm_turn"],
        stats["average_ending_turn"],
        model_name,
    )
    return table


def generate_latex_table_end_conditions(end_cond, model_name, role):
    table = r"""
\begin{table}[h!]
\centering
\begin{tabular}{l|c}
\hline
\textbf{End Condition} & \textbf{Count} \\\hline
Miscommunications & %d \\
Interceptions & %d \\
Both & %d \\
Survived & %d \\\hline
\end{tabular}
\caption{End conditions for %s as the %s in the Decrypto game.}
\label{table:end_conditions_%s}
\end{table}
""" % (
        end_cond.get("miscommunications", 0),
        end_cond.get("interceptions", 0),
        end_cond.get("both", 0),
        end_cond.get("survived", 0),
        model_name,
        role,
        role.lower(),
    )
    return table


def generate_latex_table_end_conditions_combined(
    end_cond_int, end_cond_dec, model_name
):
    table = r"""
\begin{table}[h!]
\centering
\begin{tabular}{l|c|c}
\hline
\textbf{End Condition} & \textbf{Interceptor Count} & \textbf{Decoder Count} \\\hline
Miscommunications & %d & %d \\
Interceptions & %d & %d \\
Both & %d & %d \\
Survived & %d & %d \\\hline
\end{tabular}
\caption{End conditions for %s as both the Interceptor and Decoder in the Decrypto game.}
\label{table:end_conditions_combined}
\end{table}
""" % (
        end_cond_int.get("miscommunications", 0),
        end_cond_dec.get("miscommunications", 0),
        end_cond_int.get("interceptions", 0),
        end_cond_dec.get("interceptions", 0),
        end_cond_int.get("both", 0),
        end_cond_dec.get("both", 0),
        end_cond_int.get("survived", 0),
        end_cond_dec.get("survived", 0),
        model_name.upper(),
    )
    return table


def main():
    # Load results for LLAMA as Interceptor
    llama_interceptor = load_results(
        "results/replay_experiment_human_human_llama_replay.pkl"
    )
    stats_int = llama_interceptor["aggregated_stats"]
    ir, _ = calculate_rates(stats_int)
    model_name_int = llama_interceptor["interceptor_model"]
    end_cond_int = stats_int["end_cond"]

    # Generate LaTeX table for Interceptor performance
    table_interceptor = generate_latex_table_interceptor(
        stats_int, model_name_int.upper()
    )

    # Load results for LLAMA as Decoder
    llama_decoder = load_results(
        "results/replay_experiment_human_llama_human_replay.pkl"
    )
    stats_dec = llama_decoder["aggregated_stats"]
    _, mr = calculate_rates(stats_dec)
    model_name_dec = llama_decoder["decoder_model"]
    end_cond_dec = stats_dec["end_cond"]

    # Generate LaTeX table for Decoder performance
    table_decoder = generate_latex_table_decoder(stats_dec, model_name_dec.upper())

    # Generate LaTeX table for combined end conditions
    if model_name_int == model_name_dec:
        combined_model_name = model_name_int.upper()
    else:
        combined_model_name = f"{model_name_int.upper()}/{model_name_dec.upper()}"

    table_end_conditions_combined = generate_latex_table_end_conditions_combined(
        end_cond_int, end_cond_dec, combined_model_name
    )

    # Print values for the tables
    print("LLAMA as Interceptor:")
    print(f"Total Episodes (N): {stats_int['total_episodes']}")
    print(f"Total Interceptions (I): {stats_int['total_interceptions']}")
    print(f"Interception Rate (IR%): {ir:.2f}%")
    print(f"Average Interception Turn (AIT): {stats_int['average_intercept_turn']:.2f}")
    print(f"Average Ending Turn (AET): {stats_int['average_ending_turn']:.2f}\n")

    print("LLAMA as Decoder:")
    print(f"Total Episodes (N): {stats_dec['total_episodes']}")
    print(f"Total Miscommunications (M): {stats_dec['total_miscommunications']}")
    print(f"Miscommunication Rate (MR%): {mr:.2f}%")
    print(
        f"Average Miscommunication Turn (AMT): {stats_dec['average_miscomm_turn']:.2f}"
    )
    print(f"Average Ending Turn (AET): {stats_dec['average_ending_turn']:.2f}\n")

    # Print LaTeX tables
    print("LaTeX Table for Interceptor Performance:\n")
    print(table_interceptor)
    print("\nLaTeX Table for Decoder Performance:\n")
    print(table_decoder)
    print("\nLaTeX Table for Combined End Conditions:\n")
    print(table_end_conditions_combined)


if __name__ == "__main__":
    main()
