# WebArena benchmark for BrowserGym

## Setup

1. Install the package locally for dev
```sh
cd <PATH TO THE REPO>
make install

pip install langchain
pip install langchain_openai
pip install langchain_community
```

2. Download tokenizer resources
```sh
python -c "import nltk; nltk.download('punkt')"
```

3. Setup/Reset the web servers (follow the [webarena README](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)).
```sh
BASE_URL=<YOUR_SERVER_URL_HERE>
```
You may use "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com", which is the website deployed by WebArena team for reference, for the BASE_URL for unit testing. But DO NOT use it for experiments as it can not be reset and the data will be changed by other users. Deploy your own server for experiments following the [webarena README](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md).

4. Setup the URLs as environment variables
```sh
export SHOPPING="$BASE_URL:7770/"
export SHOPPING_ADMIN="$BASE_URL:7780/admin"
export REDDIT="$BASE_URL:9999"
export GITLAB="$BASE_URL:8023"
export WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export MAP="$BASE_URL:3000"
export HOMEPAGE="$BASE_URL:4399"
```

5. Set OpenAI API key
Azure OpenAI 
```sh
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...
export OPENAI_API_VERSION=...
```

6. Run the benchmark
For some server, you need to specify the PYTHONPATH
```sh
export PYTHONPATH=<ABSOLUTE PATH TO THE REPO>
```
To run all examples in WebArena, run the following command:
```sh
cd <PATH TO THE REPO>/webarena
python test.py --model_name azureopenai/gpt-4o-2024-05-13 --headless t --use_html f --action_space bid nav stop --task_name webarena_all
```

To run a specific example in WebArena, run the following command:
```sh
python test.py --model_name azureopenai/gpt-4o-2024-05-13 --headless t --use_html f --action_space bid nav stop --task_name webarena.$TASK_ID
```

Note: 
1. Arguments can be changed as needed. But read the code to understand the arguments. The example here is the setting for current experiment.
2. If still encountering any package missing after steps before, install the package by yourself. (The whole requirements.txt will be added in the future.)

## Dev

### Augment action space
In BrowserGym, actions outputted by the model are in the format of a function call. e.g. click(123). The definition of the action functions are in core/action/functions.py.

Here are general steps to augment the action space

1. Go to core/action/functions.py. Define your action function here. 
    - If your action needs to call a external function like what "send_msg_to_user" does, you may define the external functions you need as None firstly and in the steps later, we will instruct you to define the external functions and pass it through.
    - Add proper comments as other functions do. This will be prompted to the agent as the description of the function

2. [Optional] If your function contains external functions (e.g. send_message_to_user) that have not been defined yet. Go to core/env.py and navigate to the step function in the BrowserGym class. defines the external functions here (e.g. send_message_to_user) and pass it through the execute_python_code (you may also need to go to execute_python_code and add another allowed param).

3. Go to core/actions/highlevel.py.
    - import the action function from .functions
    - group the actions and add the actions like the following example:
    ```python
    WEBARENA_EXTRA_ACTIONS = [stop]
    ```
    - in the class HighLevelActionSet, add the action_group name(will be passed through args) into ActionSubset and the corresponding part in __init__.py like the following example:
    ```python
    def __init__():
    ...
    if subsets:
            for subset in subsets:
                match subset:
                    case "chat":
                        allowed_actions.extend(CHAT_ACTIONS)
                    case "infeas":
                        allowed_actions.extend(INFEAS_ACTIONS)
                    ...
                    # add a new group
                    case "webarena":
                        allowed_actions.extend(WEBARENA_EXTRA_ACTIONS)
    ```
4. Append the group name to the args parser (if any) in your main script. Now it is in webarena/test.py:
    ```python
    parser.add_argument(
        "--action_space",
        type=str,
        nargs="+",
        default=["bid", "webarena"],
        # here the group name is "webarena" as in the previous example
        choices=["python", "bid", "coord", "nav", "webarena"],
        help="Action space to use",
    )
    ```
Now the action should work smoothly!