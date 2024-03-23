#!/bin/bash
script_dir=$(realpath $(dirname $0))
logdir='~/projects/lab02/lab2-3/log/'

install_pip_pkg() {
    pip_pkg="torch torchvision torchaudio torch_tb_profiler onnxruntime tf2onnx torchinfo"
    pip install $pip_pkg
}

stop_jupyter() {
    pkill jupyter # kill jupyter server... in brutal way XD
}

start_jupyter() {
    jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 \
                    --ip 0.0.0.0 \
                    --port 8888 \
                    --no-browser \
                    --allow-root \
                    >"${project_dir}/jupyter.stdout.log" &>"${project_dir}/jupyter.stderr.log" &
    sleep 2
    jupyter_token=$(jupyter server list | grep -E -o "*?token=[0-9a-zA-Z]+ *" | grep -E -o "=[0-9a-zA-Z]+ " | grep -E -o "[0-9a-zA-Z]+")
    echo "Token is $jupyter_token"
}

stop_tensorboard() {
    kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
}

start_tensorboard() {
    if [ ! -z $1 ]
    then
        logdir=$1
        echo "Open tensorboard w.r.t. log directory: $logdir"
    fi
    tensorboard --logdir=$logdir \
                --bind_all \
                --port=10000 \
                > tensorboard.stdout.log &> tensorboard.stderr.log &
}

case $1 in
pip_install)
    install_pip_pkg
    ;;
run_jupyter)
    stop_jupyter
    start_jupyter
    ;;
stop_jupyter)
    stop_jupyter
    ;;
run_tensorboard)
    stop_tensorboard
    start_tensorboard $2
    ;;
stop_tensorboard)
    stop_tensorboard
    ;;
*)
    echo -e "This is a script that collects useful commands for lab2 :)\nAuthor: Eroiko"
    ;;
esac
