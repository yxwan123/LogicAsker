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