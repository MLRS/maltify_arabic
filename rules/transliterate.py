# -*- coding: utf-8 -*-
import argparse
import csv
import operator
import os.path
import re
from functools import reduce
from typing import Union

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.disambig.common import Disambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize

from process import process

_VALID_DICT_FIELDS = [
    "dataset_kwargs",
]

SUN_LETTERS = "ċdnrstxżz"

_MAPPINGS = {}


def get_mappings(path: str) -> dict[str, str]:
    if path not in _MAPPINGS:
        with open(os.path.join(os.path.dirname(__file__), path), "r", encoding="utf-8") as file:
            mappings = {}
            for line in csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE):
                source = line[0] if len(line) == 2 else tuple(line[:-1])
                target = line[-1]
                mappings[source] = target
            _MAPPINGS[path] = mappings
    return _MAPPINGS[path]


_DISAMBIGUATOR = None


def _get_disambiguator(model_name: str = "msa", morphology_database: str = "calima-msa-s31", top=1) -> Disambiguator:
    global _DISAMBIGUATOR

    if _DISAMBIGUATOR is None:
        morphology_database = MorphologyDB(morphology_database) if os.path.isfile(morphology_database) else MorphologyDB.builtin_db(morphology_database)
        backoff = "ADD_PROP" if model_name in ("msa", "egy") else "NONE"
        analyzer = Analyzer(morphology_database, backoff, cache_size=100000)
        _DISAMBIGUATOR = BERTUnfactoredDisambiguator.pretrained(model_name, top=top, pretrained_cache=False)
        _DISAMBIGUATOR._analyzer = analyzer

    return _DISAMBIGUATOR


def apply_mappings(text: str, mappings_paths: list[str]) -> str:
    mappings = reduce(operator.ior, [get_mappings(file_path) for file_path in mappings_paths], {})
    transliteration = text
    for input, mapping in mappings.items():
        transliteration = transliteration.replace(input, mapping)
    return transliteration

def diacritise(tokens: list[str], disambiguator: Disambiguator, morphologically_aware: bool = True) -> list[str]:
    disambiguation = disambiguator.disambiguate(tokens)
    mappings = get_mappings("mappings/tags.tsv")
    transliterated_sequence = []
    for token_analyses in disambiguation:
        if not morphologically_aware:
            transliterated_sequence.append(token_analyses.analyses[0].diac)
            continue

        token_diacritised = []
        token_analysis = token_analyses.analyses[0].analysis
        analyses = token_analysis["bw"]
        analyses = analyses.split("+") if "+" not in token_analyses.word else [analyses]
        for analysis in analyses:
            morpheme, tag = analysis.rsplit("/", maxsplit=1)
            if morpheme == "(null)":
                continue

            if tag == "NOUN_PROP":
                morpheme = "^" + morpheme

            if "ة" in morpheme and token_analysis["stt"] == "c":
                morpheme += "~"

            morpheme = mappings.get((tag, morpheme), morpheme)
            if token_analysis["lex"] != morpheme:  # if mapping by tag only, only clitics are mapped
                morpheme = mappings.get((tag, "*"), morpheme)
            token_diacritised.append(morpheme)
        transliterated_sequence.append("".join(token_diacritised).strip())
    return transliterated_sequence


def post_process(transliteration: str) -> str:
    transliteration = re.sub(r"\^([^\s\w]*)([\w])",
                             lambda match: match.group(1) + match.group(2).upper(),
                             transliteration,
                             flags=re.UNICODE)  # capitalisation
    transliteration = re.sub(r"([aeiou]|ie)\1", r"\1", transliteration, flags=re.IGNORECASE)  # consecutive vowels
    transliteration = re.sub(r"(^|\s+)bi ħal", r"\1bħal", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)bħal il-", r"\1bħall-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)bi il-", r"\1bil-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)li il-", r"\1lill-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)fi il-", r"\1fil-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)ma' il-", r"\1mal-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)ta' il-", r"\1tal-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)għal il-", r"\1għall-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)minn il-", r"\1mill-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)bi il-", r"\1bil-", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)bil-(a|e|i|o|u|h|għ)", r"\1bl-\2", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"(^|\s+)fil-(a|e|i|o|u|h|għ)", r"\1fl-\2", transliteration, flags=re.IGNORECASE)
    transliteration = re.sub(r"l{1,2}-([lċdnrstxzż])",
                             lambda match: match.group(1).lower() + "-" + match.group(1),
                             transliteration,
                             flags=re.IGNORECASE)  # sun letter allomorphy
    transliteration = re.sub(r"\s+([.,;:?!%])", r"\1", transliteration)  # detokenisation
    return transliteration


def transliterate(sequence: Union[str, list[str]],
                  disambiguator: Disambiguator = None,
                  diacritisation: bool = True,
                  morphological_mappings: bool = True) -> Union[str, list[str]]:
    pre_tokenised = not isinstance(sequence, str)
    if pre_tokenised:
        tokens = sequence
    else:
        tokens = simple_word_tokenize(sequence)

    if diacritisation:
        if disambiguator is None:
            disambiguator = _get_disambiguator("msa", "calima-msa-s31")
        transliterated_sequence = diacritise(tokens, disambiguator, morphologically_aware=morphological_mappings)
    else:
        transliterated_sequence = tokens

    # token rules
    mappings = ["mappings/multi_character.tsv", "mappings/characters.tsv", "mappings/symbols.tsv"]
    for i, token in enumerate(transliterated_sequence):
        transliteration = apply_mappings(f"<BOS>{token}<EOS>", mappings)
        transliterated_sequence[i] = transliteration

    if not pre_tokenised:
        transliteration = " ".join(transliterated_sequence)
        transliteration = post_process(transliteration)
    else:
        for i, token in enumerate(transliterated_sequence):
            transliterated_sequence[i] = post_process(token)
        transliteration = transliterated_sequence

    return transliteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="msa",
                        help="The disambiguator model to use for the morphological analysis.",
                        )
    parser.add_argument("--morphology_database",
                        type=str,
                        default="calima-msa-s31",
                        help="The morphology database to use for the morphological analysis.",
                        )
    parser.add_argument("--no_morphological_mappings",
                        default=False,
                        action="store_true",
                        help="Whether to skip morphological mappings.",
                        )
    parser.add_argument("--no_diacritisation",
                        default=False,
                        action="store_true",
                        help="Whether to skip diacritisation. This will also skip morphological mappings.",
                        )

    process(parser,
            lambda text, args: transliterate(text,
                                             _get_disambiguator(args.model_name, args.morphology_database),
                                             diacritisation=not args.no_diacritisation,
                                             morphological_mappings=not args.no_morphological_mappings,
                                             ),
            )


if __name__ == '__main__':
    main()
