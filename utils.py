def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


def clean_and_tokenize_tweets(data):
    # Data has to be a dict containing tweets
    data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")

    # remove special characters, numbers, punctuations
    data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

    # remove words with 3 or less character, presumibily not useful
    data['tidy_tweet'] = data['tidy_tweet'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3])
    )

    # Tokenize the words for it to use
    tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())

    # Stem the words. (Stemming is a rule-based process of stripping
    # the suffixes (“ing”, “ly”, “es”, “s” etc) from a word)

    tokenized_tweet = tokenized_tweet.apply(
        lambda x: [stem(i) for i in x]
    )
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    data['tidy_tweet'] = tokenized_tweet
    return data 