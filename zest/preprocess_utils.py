# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re

from collections import Counter

def tokenize(s: str) -> list:
    return re.split(r'\W+', s.lower())

def normalize(s: str) -> str:
    s0 = re.sub(r'\b\W+ | \W+\b', ' ', s.lower())
    return re.sub(r'\b[^\w\s]+\b', '', s0)

class StringNormalizer:
    def __init__(self, no_punct: bool = False, keep_punct: bool = False):
        self.patterns_to_replace = []
        if no_punct:
            self.patterns_to_replace += [('[¿?#!¡_]+|\.\.\.+', ' ')]
        else:
            if keep_punct is False:
                self.patterns_to_replace += [
                    ('\.\.\.+', ' ___ '),
                    ('[¿?]+', ' _?_ '),
                    ('[!¡]+', ' _!_ '),
                    ('#', " _#_ "),
                ]

        self.patterns_to_replace += [
            ('[^\w\d?!_*&|#]+', ' '),
            ("(?<=\w)[’']+(?=\w)|(?<=\d)[,._\s]+(?=\d)",
             ''),  # punctuation between digits
            ('(?<!\d)\d\d?(?!\d)',
             ' ## '),  # replacing numbers with ## or ### is important
            ('(?<!\d)\d+(?!\d)', ' ### '),  # before rare word removal
            (
                '\s+',
                ' ',
            )
        ]

    def normalize_str(self, text: str):
        t = f' {text} '  # the added spaces will be removed at the end
        for p in self.patterns_to_replace:
            t = re.sub(p[0], p[1], t)
        return t[1:-1]

    def get_words(self, text: str):
        norm_text = self.normalize_str(text)
        return re.findall('\W*(\w+)', norm_text)

    def get_word_count(self, text: str):
        return len(self.get_words(text))

class CharCounter:
    def __init__(self,
                 samples: list,
                 name='',
                 IDF_thresh: float = 0.005,
                 max_samples: int = 50000):
        self.name = name
        self.IDF_thresh = IDF_thresh
        self.num_samples = min(max_samples, len(samples))
        self.populate_char_ctrs(samples[:max_samples], IDF_thresh)

        print(f"{len(self.good_chars)} good characters: ", self.good_chars)
        print(f"{len(self.leftover_chars)} leftover characters: ",
              self.leftover_chars)

        prob = self.compute_total_char_prob()
        if prob < 0.99:
            print("WARNING: In CharCounter, total probability is", prob,
                  "(should be 1.0)")

    def __str__(self):
        return f"Character counter for *{self.name}* based on {self.num_samples} samples"\
        f" and {self.num_chars} characters" \
        f"\nTop characters by DF with threshold {100 * self.IDF_thresh:.3}% " \
        f"(blank + word characters):\n {self.ctr_df_trimmed}" \
        f"\nTop characters by occurrence:\n {self.ctr_occ_trimmed}"\
        f"\nLeftover characters: {self.leftover_chars}"

    @staticmethod
    def get_char_counts(titles: list):
        ctr_df = Counter()
        ctr_occ = Counter()
        OK_chars = re.compile('[^\w\d .,:;¿?¡!_*+/|#]+')
        for t in titles:
            try:
                tn = re.sub(OK_chars, '', t)
            except TypeError:
                print(t)
                print(type(t))
                return
            ctr_df += Counter(set(tn))
            ctr_occ += Counter(tn)
        return ctr_df, ctr_occ

    def populate_char_ctrs(self, samples: list, IDF_thresh: float):
        n = len(samples)
        self.ctr_df, self.ctr_occ = self.get_char_counts(samples)
        print("Found", len(
            self.ctr_df
        ), "different characters (words, punctuation, blanks), not including emojis"
              )
        self.ctr_df_trimmed = Counter()
        self.ctr_occ_trimmed = Counter()
        for chr, count in self.ctr_df.most_common(int(
                10 / IDF_thresh)):  # limit characters to IDF > 0.005
            self.ctr_df_trimmed[chr] = count
            self.ctr_occ_trimmed[chr] = self.ctr_occ[chr]
            self.min_occ = self.ctr_occ[chr]
            if count < n * IDF_thresh: break
        self.num_chars = sum(self.ctr_occ_trimmed.values())
        self.good_chars = ''.join([c for c in self.ctr_df_trimmed])
        self.leftover_chars = ''.join(
            [c for c in self.ctr_df - self.ctr_df_trimmed])
        self.bad_chars_regex = re.compile('[^' + self.good_chars + ']+')

    def remove_bad_chars(self, s: str):
        return re.sub('\s+', ' ', re.sub(self.bad_chars_regex, '', s))

    def compute_cross_entropy(self, s: str):
        # Compute X-entropy over all characters,
        # for rare characters assume half the lowest count
        if len(s) == 0: return 0.0
        sum_lp = 0
        for c in s:
            p = self.get_char_prob(c)
            sum_lp += math.log(p)
        return -sum_lp / len(s)

    def compute_cross_entropy1(self, s: str):
        # Compute X-entropy over good characters only
        s1 = self.remove_bad_chars(s)
        if len(s1) == 0: return 0.0
        sum_lp = 0
        for c in s1:
            p = self.get_char_prob(c)
            sum_lp += math.log(p)
        return -sum_lp / len(s1)

    def get_char_prob(self, c: str):
        count = self.ctr_occ_trimmed[c]
        if count == 0:
            count = self.min_occ / 2.0
        return count / self.num_chars

    def compute_total_char_prob(self):
        # This should be close to 1.0
        sum = 0.0
        for c in self.ctr_occ_trimmed:
            sum += self.get_char_prob(c)
        return sum

class WordCounter:
    def __init__(self,
                 samples: list,
                 name='',
                 IDF_thresh: float = 0.0001,
                 max_samples: int = 50000):
        self.name = name
        self.IDF_thresh = IDF_thresh
        self.num_samples = min(max_samples, len(samples))
        self.populate_word_ctrs(samples[:max_samples], IDF_thresh)

        prob = self.compute_total_word_prob()
        if prob < 0.99:
            print("WARNING: In WordCounter, total probability is", prob,
                  "(should be 1.0)")

    def __str__(self):
        return f"Word counter for *{self.name}* based on {self.num_samples} samples"\
        f" and {self.num_chars} characters" \
        f"\nTop words by DF with threshold {100 * self.IDF_thresh:.3}% : {self.ctr_df_trimmed}" \
        f"\nTop words by occurrence:\n {self.ctr_occ_trimmed}"\

    @staticmethod
    def get_word_counts(titles: list):
        ctr_df = Counter()
        ctr_occ = Counter()
        for t in titles:
            words = t.split()
            ctr_df += Counter(set(words))
            ctr_occ += Counter(words)
        return ctr_df, ctr_occ

    def populate_word_ctrs(self, samples: list, IDF_thresh: float):
        n = len(samples)
        self.ctr_df, self.ctr_occ = self.get_word_counts(samples)
        print("Found", len(self.ctr_df), "different words")
        self.ctr_df_trimmed = Counter()
        self.ctr_occ_trimmed = Counter()
        for w, count in self.ctr_df.most_common(int(
                10 / IDF_thresh)):  # limit characters to IDF > IDF_thresh
            self.ctr_df_trimmed[w] = count
            self.ctr_occ_trimmed[w] = self.ctr_occ[w]
            self.min_occ = self.ctr_occ[w]
            if count < n * IDF_thresh: break
        self.num_words = sum(self.ctr_df_trimmed.values())
        print(f"Trimmed down to {len(self.ctr_df_trimmed)} words")

    def remove_rare_words(self, s: str):
        words = s.split()
        new_words = [w for w in words if self.ctr_df_trimmed[w] > 0]
        return ' '.join(new_words)

    def compute_cross_entropy(self, s: str):
        # Compute X-entropy over all words
        if len(s) == 0: return 0.0
        words = s.split()
        sum_lp = 0
        for c in words:
            p = self.get_word_prob(c)
            sum_lp += math.log(p)
        return -sum_lp / len(words)

    def compute_cross_entropy1(self, s: str):
        # Compute X-entropy over frequent words only
        if len(s) == 0: return 0.0
        words = s.split()
        new_words = [w for w in words if self.ctr_df_trimmed[w] > 0]
        if len(new_words) == 0: return 0.0
        sum_lp = 0
        for c in new_words:
            p = self.get_word_prob(c)
            sum_lp += math.log(p)
        return -sum_lp / len(new_words)

    def get_word_prob(self, word: str):
        count = self.ctr_occ_trimmed[word]
        if count == 0:
            count = self.min_occ / 2
        return count / self.num_words

    def compute_total_word_prob(self):
        # This should be close to 1.0
        sum = 0.0
        for word in self.ctr_occ_trimmed:
            sum += self.get_word_prob(word)
        return sum


def cleanup_str(s: str, string_normalizer=None, char_ctr=None):
    return string_normalizer.normalize_str(char_ctr.remove_bad_chars(s))

def cleanup_corpus(corpus: list, string_normalizer=None, char_ctr=None):
    return [cleanup_str(s, string_normalizer, char_ctr) for s in corpus]
