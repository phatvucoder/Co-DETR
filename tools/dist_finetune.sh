CONFIG=$1
GPUS=$2
WORKDIR=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

EXTRA_ARGS=""
if [ -f "$CONFIG" ]; then
    NUM_CLASSES=$(python3 tools/train_utils/get_nc.py "$CONFIG")

    if [ "$NUM_CLASSES" -gt 0 ]; then
        HEADS=$(python3 tools/train_utils/get_nc_configs.py "$CONFIG" "$NUM_CLASSES")
        if [ -n "$HEADS" ]; then
            EXTRA_ARGS="--cfg-options $HEADS"
        fi
    fi
fi

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir $WORKDIR $EXTRA_ARGS "${@:4}"