"""Finding Phontically similar words using edit distance

This module contains functions that find common words with nearby phonetic
pronunciations to a word given by the user.

Currently uses the Levenshtein distances (1-add, 1-delete, 2-sub) to calculate
the edit distance and finds the edit distance between the passed word and all
other words in the common word dictionary


"""


class PunWordFinder:
    """Find similar sounding words based on the CMU phoneme dictionary

    Main function takes a base word and searches through ~10k common words
    to find the ones with the smallest edit distance between the phoneme
    respresentation. In theory this should give us the words that sound most
    alike.

    Because the CMU phoneme dict is very large and full of non-useful words,
    we filter out many of them using the 10k most common english words. These
    are found in the file refered to by `GOOGLE_WORDS_FILE_NAME`

    Note:
        File names must be changed if module is called outside of the
        folder where the two files are.


    Attributes:
        MIN_LENGTH (int): minimum length of words to include in search space
        GOOGLE_WORDS_FILE_NAME (str): path of file containing common words that form our
            search space
        CMU_DICT_FILE_NAME (str): path of the file containing the CMU phoneme dictionary

    """

    MIN_LENGTH = 2
    GOOGLE_WORDS_FILE_NAME = "google10k.txt"
    CMU_DICT_FILE_NAME = "cmudict.dict"

    def __init__(self):
        """Initialize the two phonetic dictionaries we will use to find similar words

        _phonetic_dict is the entire CMU phoneme dictionary
        _filtered_dict is the 10k most common words from the CMU phoneme dictionary
        """

        self._phonetic_dict = self._get_phonetic_dict()
        self._filtered_dict = self._get_filtered_dict(self._phonetic_dict)


    def search(self, base_word):
        """Searches for similar sounding words to the passed base_word

        If the word cannot be found in the CMU phonetic dictionary we will exit
            it was likely either a rare proper noun or a misspelling

        Note:
            Base word must be all lower case (main function does this automatically)

        Args:
            base_word (str): The word to find similar sounding words of

        Returns:
            list(tuple): list of tuples where the first element is the word and the second
                is the distance from the base word
        """

        #search the larger phonetic dictonary, if not found return
        try:
            phonemes = self._phonetic_dict[base_word]
        except KeyError:
            print("Could not find word pronunciation")
            return None


        edit_distances = self._get_edit_distances(phonemes)

        return edit_distances


    def get_phonemes(self, word):
        return self._phonetic_dict[word]


    @staticmethod
    def _calc_edit_distance(base_phonemes, target_phonemes):
        """Dynamic programming technique to find edit distance between two strings

        Instead of strings of letters we will use lists of phonemes
        """

        #initialize distance matrix to all 0's
        distance_matrix = [[0 for _ in range(len(base_phonemes)+1)]
                           for _ in range(len(target_phonemes)+1)]

        for i in range(len(target_phonemes)+1):
            for j in range(len(base_phonemes)+1):
                #calculate cost to change one list into the other
                if i == 0:
                    #must add letter
                    distance_matrix[i][j] = j
                elif j == 0:
                    #must delete letter
                    distance_matrix[i][j] = i
                elif target_phonemes[i-1] == base_phonemes[j-1]:
                    #letter already in correct place, no cost
                    distance_matrix[i][j] = distance_matrix[i-1][j-1]
                else:
                    distance_matrix[i][j] = 1 + min(distance_matrix[i][j-1],
                                                    distance_matrix[i-1][j],
                                                    (distance_matrix[i-1][j-1] + 1))

        return distance_matrix[len(target_phonemes)][len(base_phonemes)]


    def _get_edit_distances(self, base_phonemes):

        edit_distances_list = list()

        for word, phonemes in self._filtered_dict.items():
            edit_distance = self._calc_edit_distance(base_phonemes, phonemes)

            edit_distances_list.append((word, edit_distance))

        #sort low to high based on edit distance
        edit_distances_list = sorted(edit_distances_list, key=lambda x: x[1])

        return edit_distances_list


    def _get_google10k_words(self):
        words = set()
        with open(self.GOOGLE_WORDS_FILE_NAME) as google_dict:
            for line in google_dict:
                word = line.strip('\n')
                if len(word) >= self.MIN_LENGTH:
                    words.add(line.strip('\n'))

        return words


    def _get_phonetic_dict(self):

        phonetic_dict = dict()
        with open(self.CMU_DICT_FILE_NAME) as cmu_dict:
            for line in cmu_dict:

                line = line.strip('\n')
                split_line = line.split()

                word = split_line[0]
                phonemes = split_line[1:]

                phonetic_dict[word] = phonemes

        return phonetic_dict


    def _get_filtered_dict(self, phonetic_dict):

        google_words = self._get_google10k_words()

        filtered_dict = dict()
        for word, phonemes in phonetic_dict.items():
            if word in google_words:
                filtered_dict[word] = phonemes

        return filtered_dict



def main():
    word_finder = PunWordFinder()

    while True:

        print("Enter base word: ", end='')
        base_word = input()
        base_word = base_word.lower()

        edit_distances = word_finder.search(base_word)

        if not edit_distances:
            #word not found, continue
            continue

        print("Searching for:", base_word)
        print("Phonemes: ", word_finder.get_phonemes(base_word))

        print()
        print('~~~~~~~~~~~~~~~~~~~~')
        print()
        print("Candidates:")
        print()

        for word, distance in edit_distances[1:6]:
            print("Word:", word)
            print("\tDistance:", distance)
            print("\tPhonemes:", word_finder.get_phonemes(word))

if __name__ == "__main__":
    main()
