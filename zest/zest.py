# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import zstandard as zstd
import re
from typing import Union, Dict, List
from preprocess_utils import normalize, tokenize

def get_hashed_words(s: str):
    # returns bytes where each word is represented by the bytes of its hash;
    # the values may change on a different invocation of python
    bytes_list = []
    for w in s.split():
        if w == '': next
        h = hash(w)
        while h > 0:
            bytes_list.append(
                1 + h % 255
            )  # 255 not 256, so that we can use 0 as a word delimiter
            h = h // 255
        bytes_list.append(0)
    return bytes(bytes_list)

class ZstdLangModel:
    # the input is assumed to be preprocessed (normalized, etc)

    ZSTD_LEVEL = 22
    dict_size_factor = 3 / 5
    verbose = 0
    max_num_dicts = 7
    min_dict_size = 150  # bytes

    def __init__(self,
                 name: str,
                 corpus: list,
                 primary_dict_size: int = 0,
                 hash_words: bool = False):
        if self.verbose > 0:
            print(f'For the "{name}" corpus of size {len(corpus)}: ')
            not_str = '' if hash_words else ' not'
            print(f'  * words are{not_str} hashed ')
        self.name = name
        self.primary_dict_size = primary_dict_size
        self.hash_words = hash_words

        self.set_params(corpus)
        self.train_dicts(corpus)

    def set_params(self, corpus: list):
        # sets `primary_dict_size` if needed

        if self.hash_words:
            self.bytes_obj_list = [get_hashed_words(s) + b'|' for s in corpus]
            corpus_b = b''.join(self.bytes_obj_list)
        else:
            corpus_b = bytes('\n'.join(corpus), 'utf8')

        orig_size = len(corpus_b)
        self.c = zstd.ZstdCompressor(level=self.ZSTD_LEVEL)
        compr_size = len(self.c.compress(corpus_b))
        del corpus_b

        if self.primary_dict_size == 0:
            self.primary_dict_size = int(self.dict_size_factor * compr_size)

        self.num_dicts = max(min(
            self.max_num_dicts,
            int(math.log2(self.primary_dict_size / self.min_dict_size))), 1)

        if self.verbose > 0:
            print(
                f'  * compression rate = {compr_size / orig_size:.4}, compr. size {compr_size}'
            )

    def train_dicts(self, corpus: list):
        samples = self.bytes_obj_list if self.hash_words else [
            bytes(s, 'utf8') for s in corpus
        ]

        def get_dict(j: int, primary_dict_size: int, samples: list):
            sz = max(int(primary_dict_size / 2**j), 256)
            return zstd.train_dictionary(sz, samples)

        if self.verbose:
            print(
                f'  * building {self.num_dicts} dictionaries of size {self.primary_dict_size} and smaller'
            )

        self.dicts = [
            get_dict(j, self.primary_dict_size, samples)
            for j in range(self.num_dicts)
        ]

        # We can skip various header info in the output since we never decompress, only check size
        self.cctx = [
            zstd.ZstdCompressor(
                dict_data=d,
                write_content_size=False,
                write_dict_id=False,
                write_checksum=False,
                level=self.ZSTD_LEVEL) for d in self.dicts
        ]

        self.header_size = len(self.cctx[0].compress(b''))
        if self.verbose > 0:
            print('  * compression headers of size', self.header_size,
                  'will be neglected')

        if self.hash_words: del self.bytes_obj_list

        return

    def get_compr_size(self, t: str):
        t_b = bytes(t, 'utf8')
        l_orig = len(t_b)
        if l_orig < 5: return l_orig, l_orig

        if self.hash_words:
            t_b = get_hashed_words(t)

        l_compr = sum([len(c.compress(t_b))
                       for c in self.cctx]) / len(self.cctx) - self.header_size
        l_orig = len(self.c.compress(t_b)) - self.header_size
        return l_compr, l_orig

class Zstd2Classifier:
    # the input is assumed to be preprocessed (normalized, etc)

    verbose = 0

    def __init__(self, name: str, pos_samples: list, neg_samples: list, hash_words : bool = False):

        if self.verbose > 0:
            print(
                f'Building a "{name}" classifier for {len(pos_samples)} positive and '
                f'{len(neg_samples)} negative samples')

        self.pos_model = ZstdLangModel('positive', pos_samples, 0, hash_words)
        self.neg_model = ZstdLangModel('negative', neg_samples,
                                       self.pos_model.primary_dict_size, hash_words)

    def score(self, s: str, num_reps : int = 2):
        # Cleanup / normalization must be performed before
        s_x = s * num_reps
        compr_sz_pos, orig_sz_pos = self.pos_model.get_compr_size(s_x)
        compr_sz_neg, orig_sz_neg = self.neg_model.get_compr_size(s_x)
        assert orig_sz_pos == orig_sz_neg, 'Two orig size estimates differ'
        if orig_sz_pos == 0 :
            return 0.0, 1.0, 1.0
        pos_rate = compr_sz_pos / orig_sz_pos
        neg_rate = compr_sz_neg / orig_sz_neg

        score = neg_rate - pos_rate
        return score, neg_rate, pos_rate

class ZstdMulticlassClassifier:
    ZSTD_LEVEL = 22
    _min_dict_size = 256 # bytes

    _version = 30 # for 0.3

    '''
    Interface - init with class samples, then call getClassAffinities() on strings
    '''

    def __init__(self, tagged_docs : Union[Dict[str, str], Dict[str, List[str]]], max_num_dicts=4):
        '''
        keys are class labels, values are text samples - either str or list[str]

        '''
        self.c = zstd.ZstdCompressor(level=self.ZSTD_LEVEL)
        self.max_num_dicts = max_num_dicts

        self.setDictSizes(tagged_docs)
        self.log(f"{len(tagged_docs)} classes;  dictionary sizes: {self.sizes}")
        self.buildDicts(tagged_docs)
        return

    def setDictSizes(self, tagged_docs : Union[Dict[str, str], Dict[str, List[str]]]) :
        '''
        Sets self.min_size, self.max_size, self.num_dicts, self.sizes
        '''
        min_size = (1 << 15)
        max_size = 0
        for tag, samples in tagged_docs.items() :
            chunks = self.text2chunks(samples)
            sz = self.getCompressedSize(" ".join(chunks))
            min_size = min(min_size, sz)
            max_size = max(max_size, sz)
        self.min_size = max(min_size // 2, self._min_dict_size)
        self.max_size = max(self.min_size, max_size)
        self.num_dicts = min(self.max_num_dicts,
                             1 + (self.max_size - self.min_size) // self.min_size)

        # when dealing with corpora of very different sizes,
        # a multiplicative formula may work better
        self.sizes = [(self.min_size + i * (self.max_size - self.min_size) // self.num_dicts)
                      for i in range(self.num_dicts)]
        return

    def buildDicts(self, tagged_docs : dict) :
        '''
        Sets self.tagged_Cdicts, self.tagged_cctx, self.header_size
        '''
        self.tagged_Cdicts = {}
        self.tagged_cctx = {}

        for tag, text in tagged_docs.items() :
            dicts = self.getCdicts(text, self.sizes)
            self.tagged_Cdicts[tag] = dicts
            self.tagged_cctx[tag] = [zstd.ZstdCompressor(
                                    dict_data=d,
                                    write_content_size=False,
                                    write_dict_id=False,
                                    write_checksum=False,
                                    level=self.ZSTD_LEVEL)
                                    for d in dicts]
        any_key = list(self.tagged_cctx.keys())[0]
        any_cctx = self.tagged_cctx[any_key][0]
        self.header_size = len(any_cctx.compress(b''))
        return

    def getCompressedSize(self, text: str) -> int :
        return len(self.c.compress(bytes(text, 'utf8')))

    def getCdicts(self, text: Union[str, List[str]], sizes: list) :
        chunks = self.text2chunks(text)
        chunks_b = [bytes(c, 'utf8') for c in chunks]
        result = []
        for sz in sizes :
            result.append(zstd.train_dictionary(sz, chunks_b))
        return result

    @staticmethod
    def text2chunks(text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            chunks = re.split(r'[.?!]\s+|\n', text) # split by punctuation and line breaks
            chunks = [normalize(t) for t in chunks if len(t) > 2]
        else:
            assert isinstance(text, list), "The input dict should have str or list[str] values"
            chunks = []
            for t in text:
                small_chunks = re.split(r'[.?!]\s+|\n', t) # split by punctuation and line breaks
                small_chunks = [normalize(t) for t in small_chunks if len(t) > 2]
                chunks.extend(small_chunks)
        return chunks

    def getClassAffinities(self, text: str, sort=True, double=True) -> list :
        chunks = self.text2chunks(text)
        t = " ".join(chunks)
        if double:
            t = t + " " + t
        text_b = bytes(t, 'utf8')
        tagged_scores = {}
        for tag, cctx_list in self.tagged_cctx.items() :
            scores = [(len(cctx.compress(text_b)) - self.header_size) / len(text_b)
                      for cctx in cctx_list]
            if len(scores) > 0:
                tagged_scores[tag] = sum(scores) / len(scores)
            else:
                tagged_scores[tag] = 0.0
        scores_list = [(1 - score, tag) for tag, score in tagged_scores.items()]
        m = min([sc for (sc, _) in scores_list])
        s = sum([sc for (sc, _) in scores_list]) - m * len(scores_list)
        if abs(s) > 1e-3 :
            scores_list = [(round((sc - m) / s, 5), tag) for (sc, tag)  in scores_list]
        if sort:
            return sorted(scores_list, reverse=True)
        return scores_list

    def log(self, message: str) :
        print(message)
        return
