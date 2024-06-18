# WebArena benchmark for BrowserGym

## Setup

1. Install the package locally for dev
```sh
pip install -r dev/requirements.txt
```

2. Download tokenizer resources
```sh
python -c "import nltk; nltk.download('punkt')"
```

3. Setup the web servers (follow the [webarena README](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)).
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
To run all examples in WebArena, run the following command:
```sh
cd webarena
python test.py --task_name webarena_all --model_name azureopenai/gpt-4o-2024-05-13 --headless t --use_html f --action_space bid webarena nav
```

To run a specific example in WebArena, run the following command:
```sh
python test.py --task_name webarena.$TASK_ID --model_name azureopenai/gpt-4o-2024-05-13 --headless t --use_html f --action_space bid webarena nav
```

Note: 
1. Arguments can be changed as needed. But read the code to understand the arguments. The example here is the setting for current experiment.
2. If still encountering any package missing after steps before, install the package by yourself. (The whole requirements.txt will be added in the future.)