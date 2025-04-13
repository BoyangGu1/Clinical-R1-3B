#!/bin/bash
API_RATE_LIMIT=10
API_KEY="FILL IN YOUR DEEPSEEK API KEY HERE"

# fill in something like ../miniconda3/bin/activate
conda_location="FILL IN YOUR CONDA LOCATION HERE"

for (( i=1; i<=API_RATE_LIMIT; i++ ))
do
    screen -dmS deepseek_api$i bash -c "source $conda_location verl; python deepseek_api_call.py --api $API_KEY --api_call_instance_index $i --api_rate_limit $API_RATE_LIMIT"
done