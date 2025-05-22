# Safeai Crew

Welcome to the Safeai

## Installation


## Clone SafeAI Github Repo

```bash
pip install poetry
git clone https://github.com/GolnooshBabaei/safeaipackage.git
cd safeai
```

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies
```bash
poetry install
```

## Set Environment Variables
```
export OPENAI_API_KEY="YOUR API KEY"
export OPENAI_API_BASE="https://api.openai.com/v1"
export CREWAI_TELEMETRY_OPT_OUT=true
export OTEL_SDK_DISABLED=true
```

## Kickoff Sample Experiment
```bash
streamlit run main.py --server.port 8080
```

Now go to http://localhost:8080