"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import argparse
from pathlib import Path

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs
import logging

import gymnasium as gym
import browsergym.webarena  # register webarena tasks as gym environments
from datetime import datetime
import json
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-4-vision-preview",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=500, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=False,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=True, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=True,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        nargs="+",
        default=["bid", "webarena"],
        choices=["python", "bid", "coord", "nav", "webarena"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.task_name == "webarena_all":
        scores = []
        for i in tqdm(range(5)):
            task_name = f"webarena.{i}"
            env_args = EnvArgs(
                task_name=task_name,
                task_seed=None,
                max_steps=30,
                headless=args.headless,
                viewport={"width": 1500, "height": 1280},
                slow_mo=args.slow_mo,
            )
            exp_args = ExpArgs(
                env_args=env_args,
                agent_args=GenericAgentArgs(
                    chat_model_args=ChatModelArgs(
                        model_name=args.model_name,
                        max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                        max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                        max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                    ),
                    flags=Flags(
                        use_html=args.use_html,
                        use_ax_tree=args.use_ax_tree,
                        use_thinking=args.use_thinking,  # "Enable the agent with thinking."
                        use_error_logs=True,  # "Prompt the agent with the error logs."
                        use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                        use_history=args.use_history,
                        use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                        use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                        use_action_history=True,  # "Prompt the agent with the action history."
                        multi_actions=args.multi_actions,
                        use_abstract_example=True,  # "Prompt the agent with an abstract example."
                        use_concrete_example=True,  # "Prompt the agent with a concrete example."
                        use_screenshot=args.use_screenshot,
                        enable_chat=False,
                        demo_mode="off",
                        action_space=args.action_space
                    ),
                ),
                logging_level=logging.INFO
            )
            exp_dir = exp_args.prepare(Path(f"./results/webarena/{task_name}"))
            exp_args.run()

            # get the reward for this task
            with open(exp_dir / "summary_info.json", "r") as f:
                summary_info = json.load(f)
            score = summary_info["cum_reward"]
            scores.append(score)

            # print pass or fail
            if score > 0:
                print(f"{task_name}: PASS")
            else:
                print(f"{task_name}: FAIL")
            
            # print current average score
            print(f"Average score: {sum(scores) / len(scores)}")
        
        # print final average score
        print(f"Final average score: {sum(scores) / len(scores)}")

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=100,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name=args.model_name,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with thinking."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=False,
                demo_mode="off",
                action_space=args.action_space
            ),
        ),
        logging_level=logging.DEBUG
    )
    exp_args.prepare(Path(f"./results"))
    exp_args.run()

if __name__ == "__main__":
    main()