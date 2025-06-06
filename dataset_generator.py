import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import gzip
import random
import re

random.seed = 74

example_pattern = re.compile("\[ESEMPIO: .*?\]")
tag_pattern = re.compile("##.*?##")


def load_compressed_json(filename):
        with gzip.GzipFile(filename, 'rb') as f:
            json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')  # Convert bytes to string
        return json.loads(json_str)

def skip_form(pos, form_thresh):
    rand_n = random.random()
    if rand_n < form_thresh:
        return True
    else:
        return False

crossword_path = "all_crosswords_definitions.json"

load_dictionary = sys.argv[1] == "True"

with open(crossword_path, "r") as f:
    data = json.load(f)

source, target, data_source = [], [], []
source_onli, target_onli, data_source_onli = [], [], []
source_neo, target_neo, data_source_neo = [], [], []

for obj in data:
    crossword_definitions = obj["crossword_definitions"].split("||")
    word = obj["word"]
    for definition in crossword_definitions:
        definition = definition.strip()
        source.append(definition)
        target.append(word)
        data_source.append("crossword")

if load_dictionary:
    data_dict = load_compressed_json("it-dictionary.gz")
    df_onli = pd.read_csv("ONLI-NEO.csv")
    df_neos = pd.read_csv("100-neos.csv")

    for w in data_dict:
        # w_clean = w.replace(" ", "")
        # w_clean = w_clean.replace("-", "")
        for pos in data_dict[w]["meanings"]:
            glossa = data_dict[w]["meanings"][pos]["glossa"]
            if "form" in pos:
                continue
            glossa = glossa.replace("\n", " ** ") # we replace the "\n" (which is the separator between senses definitions) with " ** " for easier handling
            for definition in glossa.split(" ** "):
                def_without_examples = re.sub(example_pattern, "", definition).strip()
                def_wo_tags = re.sub(tag_pattern, "", def_without_examples).strip()
                if len(def_wo_tags) > 10:
                    source.append(def_wo_tags)
                    target.append(w.lower().replace(".", ""))
                    data_source.append("dict")

    for neo, glossa in zip(df_onli["lemma"], df_onli["glossa"]):
        # neo = neo.replace(" ", "")
        # neo = neo.replace("-", "")
        source_onli.append(glossa)
        target_onli.append(neo.lower().replace(".", ""))
        data_source_onli.append("onli")

    for neo, glossa in zip(df_neos["Lemma"], df_neos["Glossa"]):
        # neo = neo.replace(" ", "")
        # neo = neo.replace("-", "")
        source_neo.append(glossa)
        target_neo.append(neo.lower().replace(".", ""))
        data_source_neo.append("neo")


df = pd.DataFrame({"source": source, "target": target, "data_source": data_source})
df_onli = pd.DataFrame({"source": source_onli, "target": target_onli, "data_source": data_source_onli})
df_neos = pd.DataFrame({"source": source_neo, "target": target_neo, "data_source": data_source_neo})
df.drop_duplicates(subset=["source"])
# rimuovere duplicati
del source
del target
del data_source

train_df, temp_df = train_test_split(df, test_size = 0.1, shuffle=True, random_state = 74)
val_df, test_df = train_test_split(temp_df, test_size = 0.5, shuffle=True, random_state = 74)

test_df = pd.concat([test_df, df_neos, df_onli]) # to keep neos only in test set

print(train_df.shape)
print(test_df.shape)
print(val_df.shape)

if load_dictionary:
    train_df.to_csv("datasets/dict_train.csv", index = False)
    val_df.to_csv("datasets/dict_val.csv", index = False)
    test_df.to_csv("datasets/dict_test.csv", index = False)
    print(train_df["data_source"].value_counts())
    print(val_df["data_source"].value_counts())
    print(test_df["data_source"].value_counts())
else:
    train_df.to_csv("datasets/train.csv", index = False)
    val_df.to_csv("datasets/val.csv", index = False)
    test_df.to_csv("datasets/test.csv", index = False)


df = pd.concat([train_df, test_df, val_df]).drop_duplicates(["target"])
df.to_csv("all_unique_words.csv", index = False)

index_dict = {}
for k, v in enumerate(df.target.values):
    index_dict[k] = v

with open("index_dict.json", "w") as f:
    json.dump(index_dict, f)