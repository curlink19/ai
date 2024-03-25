triton_version="22.12-py3-sdk"

while getopts m:v: flag
do
    case "${flag}" in
        m) model_repository=${OPTARG};;
        v) triton_version=${OPTARG};;
    esac
done

echo "triton_version: $triton_version"
echo "model_repository: $model_repository"

docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $model_repository:/models nvcr.io/nvidia/tritonserver:$triton_version tritonserver --model-repository=/models