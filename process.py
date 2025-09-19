import abc
import argparse
import csv
import json
import os
from typing import Callable, Sequence, Union, TextIO

from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.stringutils import force_unicode
from datasets import load_dataset
from tqdm import tqdm

_VALID_DICT_FIELDS = [
    "dataset_kwargs",
]


def _convert_str_dict(passed_value: dict):
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="The name of the dataset to use (via the datasets library).",
                        )
    parser.add_argument("--dataset_config_name",
                        type=str,
                        default=None,
                        help="The configuration name of the dataset to use (via the datasets library).",
                        )
    parser.add_argument("--train_file",
                        type=str,
                        default=None,
                        help="The path to the training data file (a jsonlines or csv file).",
                        )
    parser.add_argument("--validation_file",
                        type=str,
                        default=None,
                        help="The path to the validation data file (a jsonlines or csv file).",
                        )
    parser.add_argument("--test_file",
                        type=str,
                        default=None,
                        help="The path to the testing data file (a jsonlines or csv file).",
                        )
    parser.add_argument("--dataset_kwargs",
                        type=Union[str],
                        default="{}",
                        help="Extra parameters to use when loading the dataset."
                        )
    parser.add_argument("--text_column",
                        type=str,
                        default="text",
                        help="The name of the column in the datasets containing the texts.",
                        )
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="The path to the directory where the output transliterations are persisted. "
                             "Output files will be `.txt` files for each available split in the dataset.",
                        )
    args = parser.parse_args()

    for field in _VALID_DICT_FIELDS:
        passed_value = getattr(args, field)
        # We only want to do this if the str starts with a bracket to indicate a `dict`
        # else it's likely a filename if supported
        if isinstance(passed_value, str) and passed_value.startswith("{"):
            loaded_dict = json.loads(passed_value)
            # Convert str values to types if applicable
            loaded_dict = _convert_str_dict(loaded_dict)
            setattr(args, field, loaded_dict)

    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.test_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")

    return args


mapper = CharMapper.builtin_mapper("arclean")


def normalise_arabic(text: str) -> str:
    text = normalize_unicode(text)
    text = force_unicode(text)
    text = mapper.map_string(text)
    text = dediac_ar(text)

    return text


class Writer(abc.ABC):

    def __init__(self, text_field: str):
        self.text_field = text_field

    def write(self, data, processed_text: Union[str, Sequence[str]], tokenised: bool = False) -> None:
        data[self.text_field] = processed_text
        self._write(data, tokenised)

    @abc.abstractmethod
    def _write(self, data: dict, tokenised: bool = False) -> None:
        pass

    @staticmethod
    def build(file_extension: str, text_field: str, file: TextIO):
        if file_extension in ("csv", "tsv"):
            return CsvWriter(text_field, file, delimiter="\t" if file_extension == "tsv" else ",")
        elif file_extension in ("json", "jsonl"):
            return JsonWriter(text_field, file)
        else:
            return TextWriter(file)


class CsvWriter(Writer):

    def __init__(self, text_field: str, file: TextIO, delimiter: str = ","):
        super().__init__(text_field)
        self.writer = csv.writer(file, delimiter=delimiter, lineterminator="\n")

    def _write(self, data: dict, tokenised: bool = False) -> None:
        if not tokenised:
            self.writer.writerow(data.values())
        else:
            length = len(next(iter(filter(lambda value: isinstance(value, list), data.values()))))
            data = {key: value if isinstance(value, list) else [value] * length for key, value in data.items()}
            for values in zip(*list(data.values())):
                self.writer.writerow(values)


class JsonWriter(Writer):

    def __init__(self, text_field: str, file: TextIO):
        super().__init__(text_field)
        self.file = file

    def _write(self, data: dict, tokenised: bool = False) -> None:
        json.dump(data, self.file, ensure_ascii=False)
        self.file.write("\n")


class TextWriter(Writer):

    def __init__(self, text_field: str, file: TextIO):
        super().__init__(text_field)
        self.file = file

    def _write(self, data: dict, tokenised: bool = False) -> None:
        if tokenised:
            output = data[self.text_field]
        else:
            output = "\n".join(data[self.text_field]) + "\n"
        self.file.write(output + "\n")


def process(parser: argparse.ArgumentParser, processor: Callable[[Union[str, Sequence[str]], argparse.Namespace], Union[str, Sequence[str]]]):
    args = parse_args(parser)

    data_files = {}
    extension = "txt"
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            data_files=data_files or None,
            # cache_dir=args.cache_dir,
            # token=args.token,
            # trust_remote_code=args.trust_remote_code,
        )
    else:
        dataset_kwargs = {}
        if extension == "jsonl":
            builder_name = "json"  # the "json" builder reads both .json and .jsonl files
        elif extension == "tsv":
            builder_name = "csv"  # the "csv" builder reads both .csv and .tsv files
            dataset_kwargs = {"delimiter": "\t"}
        else:
            builder_name = extension  # e.g. "parquet"
        dataset = load_dataset(
            builder_name,
            data_files=data_files,
            # cache_dir=args.cache_dir,
            # token=args.token,
            **dataset_kwargs,
            **args.dataset_kwargs,
        )

    os.makedirs(args.output_path, exist_ok=True)
    for split, dataset_split in dataset.items():
        with open(os.path.join(args.output_path, f"{split}.{extension}"), "w", encoding="utf-8") as file:
            writer = Writer.build(extension, args.text_column, file)
            for instance in tqdm(dataset_split, desc=f"Processing {split} split"):
                text = instance[args.text_column]
                pre_tokenised = not isinstance(text, str)
                if pre_tokenised:
                    text = [normalise_arabic(token) for token in text]
                else:
                    text = normalise_arabic(text)

                processed_text = processor(text, args)

                writer.write(instance, processed_text)
