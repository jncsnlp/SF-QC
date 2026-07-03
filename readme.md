# SF-QC: Explore the Selection Bias of Large Language Models in Zero-Shot Out-of-Distribution Intent Detection
![architecture](https://github.com/jncsnlp/SF-QC/blob/main/framework.png)

This repository contains the official PyTorch implementation code for SF-QC: Explore the Selection Bias of Large Language Models in Zero-Shot Out-of-Distribution Intent Detection: <a href="https://ieeexplore.ieee.org/document/11523681">SF-QC</a>.

## Installation

First, clone the repository locally:

```
git clone https://github.com/jncsnlp/SF-QC.git
cd SFQC
```

## Requirements

#### Install required packages

```
conda create -n SFQC python=3.9 -y
codna activate SFQC
pip install -r requirement.txt
```

#### Download models

For the model used to calculate the similarity between intent and text, we employ **all-MiniLM-L6-v2**, which can be downloaded from [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

For the inference model, we use [Qwen2.5-7B](https://github.com/QwenLM/Qwen), [DeepSeek-R1-7B](https://github.com/deepseek-ai/DeepSeek-R1) and [Llama3-8B](https://github.com/meta-llama/llama3), and we utilize [Ollama](https://github.com/ollama/ollama) to deploy and run these models. For installation steps of Ollama, please consult its official website: [Linux - Ollama](https://docs.ollama.com/linux#manual-install). 

After installing Ollama, you can use the following commands to install the Qwen2.5-7B model, for other models, you can search from [Model Lists](https://ollama.ac.cn/library).

```
ollama pull qwen2.5:7b
```

## Usage

Please start the Ollama service in another terminal before running.

```
ollama serve
```

Run example:

```
bash run.sh
```

or

```
python scripts/run.py \
  --task [TASK] \
  --id_ratio [ID_RATIO] \
  --llm [MODEL] \
  --method [METHOD]
# [TASK] can be: banking, clinic
# [ID_RATIO] can be: 0.25, 0.5, 0.75
# [MODEL] can be: qwen, deepseek, llama
# [METHOD] can be: SF-QC, CoT
```

Evaluate example:

```
bash eval.sh
```

or

```
python scripts/evaluate.py \
    --input_file [RESULT_PATH] \
    --type [TYPE] \
    --show_class_accuracy \
    --output-json [OUTPUT_PATH]
# RESULT_PATH: your results path
# TYPE can be: banking25, banking50, bangking75, clinc25, clinc50, clinc75
# OUTPUT_PATH: the path you want to save the evaluation result
```

## Acknowledgement

Our SF-QC method is built upon [Ollama](https://github.com/ollama/ollama) and three open-source large language models ([Qwen2.5-7B](https://github.com/QwenLM/Qwen), [DeepSeek-R1-7B](https://github.com/deepseek-ai/DeepSeek-R1) and [Llama3-8B](https://github.com/meta-llama/llama3)), and we gratefully acknowledge their contributions.

## Cite

```
@article{lu2026sf,
  title={SF-QC: Explore the Selection Bias of Large Language Models in Zero-Shot Out-of-Distribution Intent Detection},
  author={Lu, Heng-yang and Liu, Xin-yi and Zhang, Jia-ming and Fan, Chenyou and Fang, Wei},
  journal={IEEE Transactions on Computational Social Systems},
  year={2026},
  publisher={IEEE}
}
```

Contact: liuxinyi@stu.jnu.edu.cn

If you encounter any difficulties or problems while using our code, please feel free to contact us. If you find our paper or code helpful, please give us a like. ❤️
