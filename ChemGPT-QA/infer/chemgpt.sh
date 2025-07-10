conda activate braingpt
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server --served-model-name chemgpt --model /path/to/chemgpt --gpu-memory-utilization 0.98 --tensor-parallel-size 4 --port 6001
