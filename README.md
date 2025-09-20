# Maltification of Arabic

This repository contains tools for processing Arabic data through transliteration with a Maltese-centric focus.

## Arabic Data Processing

A [small demo](demo.ipynb) is provided to apply transliteration of Arabic data with different methods.

Otherwise, to transliterate an entire dataset, follow these steps:
1. If the source data is in a different format than the target data, reformat it as the target data.
   The data is stored in some `$INPUT_DIRECTORY`.
2. Clean the text using [arclean](https://camel-tools.readthedocs.io/en/latest/api/utils/charmap.html#utility).
   _The transformation code does this automatically as well, but this step is necessary if you are not transforming the
   source Arabic data further in the following steps._
3. Transform the data from above with the [appropriate method](#arabic-data-transformation).
   This step will generate a file for each split in the dataset in some `$OUTPUT_PATH`.

### Arabic Data Transformation

Scripts are provided to process an entire dataset with accordingly.
The following arguments are shared across scripts:
- The input data, which is compatible with [Hugging Face `datasets`](https://www.github.com/huggingface/datasets/).
   When passing `--train_file` (& `--validation_file`/`--test_file`), these should be JSON/CSV files.
- A `--text_column` argument, specifying the field to transform, ignoring the rest of the data.
- An `--output_path` argument to specify in which directory the transformed data is persisted.

The following table provides a breakdown of the arguments to be specified for each dataset.

| Dataset            | `$INPUT_DATA_ARGS`                                                                                             | `$TEXT_COLUMN` |
|--------------------|----------------------------------------------------------------------------------------------------------------|----------------|
| Sentiment Analysis | `--train_file="$DATA_PATH/sentiment_analysis/train.csv" --dataset_kwargs="{\"names\": [\"label\", \"text\"]}"` | `"text"`       |
| ANERCorp           | `--train_file="$DATA_PATH/ANERcorp-CamelLabSplits/train.json"`                                                 | `"tokens"`     |

These arguments are referred to as `$DATASET_ARGS` below, and have the following format: `$INPUT_DATA_ARGS --text_column=$TEXT_COLUMN --output_path=$OUTPUT_PATH`.
Further details on how to transform the data with each method is given below:

<details>
<summary>MorphTx</summary>
To run the full system with diacritisation and morpheme-level mappings:

```shell
python rules/transliterate.py $DATASET_ARGS \
    --model_name="egy" --morphology_database="$CAMEL_TOOLS_PATH/data/morphology_db/calima-egy-c044.db"
```
</details>

<details>
<summary>CharTx</summary>
To run the simple system with only character-level mappings:

```shell
python rules/transliterate.py $DATASET_ARGS \
    --no_diacritisation
```
</details>

<details>
<summary>Buckwalter</summary>

To run Buckwalter (lower-cased):

```shell
python baseline/transliterate.py $DATASET_ARGS \
    --scheme=buckwalter
```
</details>

<details>
<summary>Uroman</summary>

To run Uroman:

```shell
python baseline/transliterate.py $DATASET_ARGS \
    --scheme=uroman
```
</details>


## Citation

This work was introduced in [Data Augmentation for Maltese NLP using Transliterated and Machine Translated Arabic Data](https://arxiv.org/abs/2509.12853).
Cite as follows:

```bibtex
@inproceedings{micallef-etal-2025-maltification,
    title={Data Augmentation for Maltese NLP using Transliterated and Machine Translated Arabic Data}, 
    author={Kurt Micallef and Nizar Habash and Claudia Borg},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    eprint={2509.12853},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.12853}, 
}
```
