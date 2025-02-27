# GUIPivot.Private

<div align="center">
  <h2 align="center">Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks</h2>
  <!-- <a href="https://arxiv.org/abs/2412.02454" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.02454-b31b1b.svg?style=flat">
  </a> -->
</div>

**This repository is the code implementation of our paper:**
```
Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks
```
We propose a pivot task name **query inference** to smooth coordinate-oriented grounding and action-oriented reasoning, imporoving the performance of MLLM-powered GUI agents in resource-constrained scenarios.
<div align="center">
  <a style="display: inline-block; text-align: center;">
      <img src="./doc_images/demo.png">
  </a>
</div>


## 🆙 Updates
* [x] We release the code for data preprocess, query refinement, and evaluation.
* [x] We release the training data in sharegpt format.

## Dependencies
- The training for grounding, query inference, and reasoning is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), refer the LLaMA-Factory repo for training environment installation. 
- After setting up LLaMA-Factory for training, you can install other requirements by:
```bash
    pip3 install -r requirements.txt
```

## Training Data
- We release the json training data in LLaMA-Factory-supported shargpt format in [./data/], please unzip [./data/GUI.zip](./data/GUI.zip) by the following command to obtain the json data along with [dataset_info.json](./data/dataset_info.json) for LLaMA-Factory training. UIBERT for query inference is also provided in [./data/GUI/uibert_query_summary_polished_preprocessed_train.json](./data/GUI/uibert_query_summary_polished_preprocessed_train.json)
```bash
    zip -s 0 GUI.zip --out combined.zip
    unzip combined.zip
```
- The original datasets can be downloaded as follows:
  - **UIBERT**: UIBERT is a subset of the [OS-Atlas](https://hf-mirror.com/datasets/OS-Copilot/OS-Atlas-data) grounding dataset. Refering the above link for downloading.
  - **AITZ**: [AITZ](https://github.com/IMNearth/CoAT) is a mobile agent dataset derived from a subset of AITW and annotated by proprietary MLLMs for CoAT components. Refering the above link for downloading.
  - **AndroidControl**: [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) is a mobile agent dataset comprising 15,283 demonstrations with step-wise instructions. Refering the above link for downloading the tf record file and extracting the image file.
- After downloading the original datasets, adjust the file folder structure:
  - **UIBERT**: Moving all images to [./GUIData/UIBert/screenshots](./GUIData/UIBert/screenshots).
  - **AITZ**: Moving all images to [./GUIData/android_in_the_zoo/train/images](./GUIData/android_in_the_zoo/train/images) and [./GUIData/android_in_the_zoo/test/images](./GUIData/android_in_the_zoo/test/images) accroding to the original split setting.
  - **AndroidControl**: Moving the original dataset to [./GUIData/android_control_parsed](./GUIData/android_control_parsed) and run the following command for processing.
```bash
    python mergeAndroidControl.py
```
- After adjusting, the file folder structure should be:
```
.
└── GUIData
    ├── UIBert
    │   ├── screenshots
    │   │   └── image_files
    │   └── uibert_raw.json
    ├── android_control
    │   ├── images
    │   │   └── image_files
    │   ├── jsons
    │   │   └── json_files
    │   └── layouts
    │       └── accessibility_tree_files
    └── android_in_the_zoo
        ├── test
        │   ├── images
        │   │   └── image_files
        │   └── jsons
        │       └── json_files
        └── train
            ├── images
            │   └── image_files
            └── jsons
                └── json_files
```

## Constructing Dataset for Query Inference

<div align="center">
  <a style="display: inline-block; text-align: center;">
      <img src="./doc_images/queryInferencePipeline.png">
  </a>
</div>


We provide the code implementation for constructing UIBERT for query inference. Detailed process are provided as follows:

1. Prepare the original UIBERT dataset and adjust file folder structure as mentioned above.
2. Configuring API key for **Qwen-VL-Max** in [refineQuery.py](refineQuery.py):
```python
client = OpenAI(
    api_key="your key", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```
3. Run `python refineQuery.py` for constructing.

Example triplets $\langle s, q_r, c \rangle$ from the refined UIBERT dataset, along with the original query $q$ are provided as follows:
<div align="center">
  <a style="display: inline-block; text-align: center;">
      <img src="./doc_images/uibertSamples.png">
  </a>
</div>

## Evaluation
After acquiring the fine-tuned model for reasoning, configure and run `evaluate.py` to obtain the evaluation results on two mobile agnet benchmarks.

Detailed arguments setting:
```bash
python evaluate.py \
    [--type:acactioneval for AndroidControl and aitzablationactioneval for AITZ] \
    [--testJsonPath: the test split dataset json path] \
    [--log: the result record path] \
    [--modelPath: the transformer model path for fine-tuned reasoning model] \
```
# Acknowledgement
This work can not be done without the help of the following repos:


- LLaMA-Factory: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

<!-- # Citation

```ruby
@atricle{wu2025guipivot,
  title   = {Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks},
  year    = {2025}
}
``` -->
