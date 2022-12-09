# Paper Analyzer

Repository for academic paper analyzer. 
From abstract of conference paper, we generate TL;DR, strength, weakness. 
Also, we predict if the paper is accepted or not from the paper abstract.

## How to run

### Requirements
This project is tested under:
- python 3.8
- python 3.9

To install the requirements, first fix 

```text
--extra-index-url https://download.pytorch.org/whl/cu116
torch==1.12.1
```
to CUDA version appropriate for your system. Then run

```shell
pip install -r requirements.txt
```
under your environment.
The project is tested under virtualenv and conda.

### Data crawling
We collected paper data from [openreview](https://openreview.net/). \\
We use openreview API for data collection. We collect data from all possible venues and years. 
To collect the data from openreview, run:

```shell
python crawl.py [--weakness TRUE/FALSE]
```
args:
- weakness: collect weakness or not

We provide collected data in `crawled` directory.

### Train

To run training TL;DR/Strength/Weakness generation, run

```shell
python train.py --task [TASK]
```
TASK includes tldr(TL;DR), strength, and weakness.

To train acceptance predictor, run

```shell
python train_test_accept.py
```

### Test
To test generation of each analyses from abstract, run

```shell
python generate.py --task TASK --gen_num NUM
```
TASK includes tldr(TL;DR), strength, and weakness.

NUM is the number of generated sentences (TL;DRs, Strengths, Weaknesses) to generate.

To test the acceptance predictor, run
```shell
python train_test_accept.py --test_only
```

### Project Demo
You can run demo and analyze your own abstract. 
After completing the train, run
```shell
streamlit run app.py
```
By default, you can check project in following address
- http://[Server IP]:8501

## To-do Lists
- [x] Crawl more conferences - All venues including conferences, workshops, tutorials, tracks were crawled!
- [x] Make dataset - generate review
    - [x] TL;DR
    - [x] Overview (Summary of review)
    - [x] Strength
    - [x] Weakness
- [x] Make dataset - paper related
    - [x] Abstract
- [ ] Report descriptive statistics of dataset
    - The total number of accepted and rejected papers for each conference
    - The number of papers that contain strength and weakness 
    - The number of sentences (?)
