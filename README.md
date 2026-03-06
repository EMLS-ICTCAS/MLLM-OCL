# MLLM-OCL: A Online Continue Learning Framework for Multimodal Large Language Models (The winnner of 6th CLVision Workshop @ ICCV 2025 Challenge)

This is the official code for MLLM-OCL: A Online Continue Learning Framework for Multimodal Large Language Models

We are the team  Team ICT-EMLS, which is the [winner] (https://sites.google.com/view/clvision2025/challenge/challenge-results) of 6th CLVision Workshop @ ICCV 2025 Challenge.

For more details about the 6th CLVision Workshop @ ICCV 2025 Challenge, please refer to the [challenge website](https://sites.google.com/view/clvision2025/challenge).

## Method explanation
The explanation of our method is stored in [./explanation](./explanation), including the [document](./explanation/explanation.pdf), the [PPT](./explanation/explanation.pptx), and the [PPT presentation](./explanation/explanation.mp4).

## How to run the code
The recommended setup steps are as follows:
1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Clone the repo and create the conda environment
```bash
git clone https://github.com/EMLS-ICTCAS/MLLM-OCL
cd MLLM-OCL
conda create -n MLLM-OCL python=3.10
conda activate MLLM-OCL
bash env_setup.sh
bash prepare.sh # model checkpoint (~15GB)
```
3. Download the competition data: download the data for the scenarios from [google drive](). We recommand using [gdrive](https://github.com/glotlabs/gdrive) to download the data.
```bash
mkdir dataset
cd dataset
gdrive files download 1au5I-ZYBDiEG5xyLliU49Sb7iddOikAf # Upstream Scenario 1 datasets (1.5GB)
tar -xvf Upstream_scenario1.tar
rm Upstream_scenario1.tar
gdrive files download 1JdT13s3ycHuBo-zC_vspTfifgwxB-ual # Upstream Scenario 2 datasets (11GB)
tar -xvf Upstream_scenario2.tar
rm Upstream_scenario2.tar
gdrive files download 1ajHvpKBgoj3Kg6BX8EAXsPub-dMP3KLY # Downstream datasets (9GB)
tar -xvf Downstream.tar
rm Downstream.tar
cd ..
```

4. Run Continual training on upstream tasks & fine-tuning on downstream tasks
```bash
conda activate clvision
bash adapt.sh 1 # run scenario 1
bash adapt.sh 2 # run scenario 2
```
Checkpoint will be saved in `checkpoints_${NOTE}`

5. Run evaluation for submission

Please set `NOTE` in `eval.sh` and `create_submission_file.py` to `NOTE` in `adapt.sh`
```bash
bash eval.sh 1 # evaluate scenario 1
bash eval.sh 2 # evaluate scenario 2
python create_submission_file.py
```

6. Submit the generated `submission.zip` file to the [competition leaderboard](https://www.codabench.org/competitions/9421/).

## Code rule
- Implement your strategy in python files in `strategies` folder. (You can still modify some codes in other folders.)
- Please do not modify the parts wrapped with
```bash
# ==================================== DO NOT MODIFY THIS PART ===================================#
...
# ================================================================================================#
```

