import argparse
import re
from typing import Callable, Sequence, Union

import uroman
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.charsets import AR_CHARSET
from camel_tools.utils.transliterate import Transliterator

from process import process

buckwalter_transliterator = Transliterator(CharMapper.builtin_mapper("ar2bw"))


def buckwalter_transliterate(text: str, lowercase=True) -> str:
    transliteration = buckwalter_transliterator.transliterate(text)
    if lowercase:
        # uppercase in Buckwalter has a different meaning than Latin uppercasing
        transliteration = transliteration.lower()
    return transliteration


uroman_transliterator = uroman.Uroman()


def uroman_transliterate(text: str) -> str:
    return uroman_transliterator.romanize_string(text, "ara")



SCHEMES = {
    "buckwalter": buckwalter_transliterate,
    "uroman": uroman_transliterate,
}


def transliterate(sequence: Union[str, Sequence[str]], transliterator: Callable[[str], str]) -> Union[str, Sequence[str]]:
    pre_tokenised = not isinstance(sequence, str)
    if pre_tokenised:
        tokens = sequence
    else:
        tokens = simple_word_tokenize(sequence)

    transliteration = []
    for token in tokens:
        if re.match(rf"[{AR_CHARSET}]", token):
            output = transliterator(token)
        else:
            output = token
        transliteration.append(output)

    if pre_tokenised:
        return transliteration
    else:
        return re.sub(r"\s+([.,;:?!%])", r"\1", " ".join(transliteration))  # detokenisation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme",
                        choices=SCHEMES.keys(),
                        default=next(iter(SCHEMES.keys())),
                        help="The transliteration system to use")

    process(parser, lambda text, args: transliterate(text, SCHEMES[args.scheme]))


if __name__ == '__main__':
    main()
