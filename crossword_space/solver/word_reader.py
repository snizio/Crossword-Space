from . import DATA_DIR


class WordReader:
    @staticmethod
    def read(name):
        word_list = []
        with open(DATA_DIR / "wordlists" / f"{name}.txt") as my_file:
            for line in my_file:
                line = line.strip()
                word_list.append(line)
        return word_list
