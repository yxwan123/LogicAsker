function clean_up(){
  kill $@
  sleep 1
  kill -9 $@
  rm -rf *.log
}

if [ "$1" = "env" ]
then
    conda init
    pip install fschat
    pip install pydantic==1.10.11
    pip install --upgrade openai
    pip install --upgrade GoogleBard
fi


if [ "$1" = "vicuna" ]
then
    export HUGGINGFACE_HUB_CACHE=~/autodl-tmp/cache
    python -m fastchat.serve.controller --host 127.0.0.1 &
    PID1=$!
    python -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.3 --host 127.0.0.1 &
    PID2=$!
    python -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8000 &
    PID3=$!
    
    trap "clean_up $PID1 $PID2 $PID3" EXIT
    wait $PID1
fi
