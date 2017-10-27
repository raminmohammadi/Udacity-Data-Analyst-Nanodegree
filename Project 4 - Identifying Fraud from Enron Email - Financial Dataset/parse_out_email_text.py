import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 

        
        """
    from nltk.stem.snowball import SnowballStemmer

    f.seek(0)  ## go back to beginning of file (annoying)
    all_text = f.read()

    ## split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        translate = str.maketrans("", ""), string.punctuation
        text_string = content[1].translate(translate)


        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)


        stemmer = SnowballStemmer("english")
        stem_words = []

        for i in text_string.split():
            stem_words.append(stemmer.stem(i))

        text_string = " ".join(stem_words)

        words = text_string
        
    return words
    

# wrap it in main
def main():
    ff = open("../txt_Files/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)

if __name__ == '__main__':
    main()
