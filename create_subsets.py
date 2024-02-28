import pandas as pd
from nltk.tokenize import word_tokenize

target_words = ['gelingen', 'gelingt', 'gelungen', 'schaffen', 'geschafft', 'durchkommen', 'bewältigen',
         'erledigt', 'erledigen', 'bewerkstelligen', 'bewerkstelligt', 'meistern', 'meistert', 'gemeistert', 'erreichen', 'erreicht']

def make_german_subset(in_filename, out_filename):
    with open(in_filename, 'r') as de:
        with open(out_filename, 'w') as out:
            de_sents = de.read().splitlines()
            for sent in de_sents:
                if any(word in sent for word in set(target_words)):
                    out.write(sent + '\n')


def create_other_subsets(target_english_file = 'europarl/fi-en.txt', target_file = 'europarl/finnish.txt', output_subset_file = 'fi_subset.txt'):
    # with open('europarl/deutsch.txt', 'r') as de:
    # with open('europarl/english.txt', 'r') as en:
    #        en_sents = en.read().splitlines()
    with open(target_english_file, 'r', encoding="utf8") as target_en:
        target_en_sents = target_en.read().splitlines()
    with open('en_subset.txt', 'r', encoding="utf8") as en:
        en_sents = en.read().splitlines()
    with open(target_file, 'r', encoding="utf8") as target:
        target_sents = target.read().splitlines()
    # write subset file if match between the two english sentences
    with open(output_subset_file, 'w') as output_file:
        for goal_sent in en_sents:
            for en, fi in zip(target_en_sents, target_sents):
                if goal_sent == en:
                    try:
                        output_file.write(fi + '\n')
                    except UnicodeEncodeError:
                        print('Unicode Error!')
                        continue

    en_to_target_map = dict(zip(target_en_sents, target_sents))

    return en_to_target_map

def write_matching_file(en_sents, en_to_target_map):
    with open('fi_subset.txt', 'w') as es_file:
        for goal_sent in en_sents:
            fi_sent = en_to_target_map.get(goal_sent, None)
            if fi_sent is not None:
                es_file.write(fi_sent + '\n')
            else:
                es_file.write('<no match>' + '\n')



if __name__ == '__main__':
    make_german_subset('europarl/deutsch.txt', 'test_outfile.txt')
    create_other_subsets('europarl/english.txt')
    df = pd.read_pickle('lextyp_pickle_clean.pkl')

    df['DE_tokenized'] = df['DE'].apply(lambda x: ' '.join(word_tokenize(str(x))))
    df['EN_tokenized'] = df['EN'].apply(lambda x: ' '.join(word_tokenize(str(x))))
    df['ES_tokenized'] = df['ES'].apply(lambda x: ' '.join(word_tokenize(str(x))))
    df['FI_tokenized'] = df['FI'].apply(lambda x: ' '.join(word_tokenize(str(x))))


