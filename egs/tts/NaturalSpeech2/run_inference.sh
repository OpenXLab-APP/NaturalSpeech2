######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config.json"
exp_name="ns2_libritts"
ref_audio="$work_dir/egs/tts/NaturalSpeech2/prompt_example/ref_audio.wav"
checkpoint_path="$work_dir/ckpts/tts/ns2_libritts/checkpoint/epoch-0065_step-0376136_loss-7.126379"
output_dir="$work_dir/output"
mode="single"

export CUDA_VISIBLE_DEVICES="0"

######## Parse Command Line Arguments ###########
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --text)
    text="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

######## Train Model ###########
python "${work_dir}"/bins/tts/inference.py \
    --config=$exp_config \
    --text="$text" \
    --mode=$mode \
    --checkpoint_path=$checkpoint_path \
    --ref_audio=$ref_audio \
    --output_dir=$output_dir \