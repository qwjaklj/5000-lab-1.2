import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data (
    ronald_reagan and barack_obama) and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        path = Path(folder)
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f) as fp:
                text = fp.read()
                # TODO If the directory name is either kennedy or johnson,
                #  create a pandas DataFrame where the column "authors" is
                #  the directory name and there is a field "text" which
                #  contains the text from the opened file. Note that you want
                #  a single DataFrame, but you loop over numerous files.
                if dir_name in ("kennedy", "johnson"):
                    new_df = pd.DataFrame({"author": [dir_name], "text": [text.lower()]})
                    df = pd.concat([df,new_df],ignore_index=True)
                else:
                    # TODO Otherwise, we want to create a DataFrame for the
                    #  unlabeled data in a similar fashion. But this is a
                    #  little different because we don't get the label from
                    #  the directory but instead from the file name. Again,
                    #  the field "author" should have the author's name and
                    #  the field "text" should contain the text.
                    new_df = pd.DataFrame({"author": [f.stem.split('_')[-1]], "text": [text.lower()]})
                    df = pd.concat([df,new_df],ignore_index=True)
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))
    return df_train, df_test


def train_nb(df, alpha=0.1):
    """
    Takes as input a pandas DataFrame containing Federalist
    files text to determine priors and likelihoods
    :param df: a pandas DataFrame
    :return: two numpy arrays for the priors and likelihoods
    """
    # TODO Create a dictionary that maps whitespace-separated tokens in the
    #  file to a unique index. Also, create variables for the number of
    #  documents and the number of classes. Use df.shape for these.
    vocabulary = {word: i for i, word in enumerate(set(' '.join(df['text']).split()))}
    n_docs = df.shape[0]
    n_classes = len(set(df['author']))
    # TODO Compute the priors
    priors = np.array([np.sum(df['author'] == i)/n_docs for i in range(n_classes)])
    # TODO Create a matrix containing all 0s called training_matrix of size
    #  (n_docs, len(vocabulary)), then fill it with the counts of each word
    #  for each document. This is the bag-of-words matrix for all the documents
    training_matrix = np.zeros((n_docs, len(vocabulary)))
    for i, text in enumerate(df['text']):
        for word in text.split():
            if word in vocabulary:
                training_matrix[i, vocabulary[word]] += 1
    # TODO Get word counts for both classes
    word_counts_per_class = np.zeros((n_classes,len(vocabulary)))
    for c in range(n_classes):
        word_counts_per_class[c] = np.sum(training_matrix[df['author']==c],axis=0) 
    # TODO Initialize a matrix to store the likelihoods
    likelihoods = np.zeros(((n_classes,len(vocabulary))))
    for i in range(n_classes):
        likelihoods[i] = (word_counts_per_class[i] + alpha) / (np.sum(word_counts_per_class[i]) + (len(vocabulary)+1) * alpha)
    return vocabulary, priors, likelihoods


def test(df, vocabulary, priors, likelihoods):
    """
    Takes as input a pandas DataFrame representing the disputed Federalist
    Papers and returns predictions for every text document
    :param df: a pandas DataFrame
    :return: a numpy array of predictions
    """
    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))
        # TODO Fill test_vector with counts for the words that appear in the
        #  vocabulary
        for word in text.split():
            if word in vocabulary:
                test_vector[vocabulary[word]] += 1
        # TODO Compute predictions p(y|test)
        preds = np.log(priors) + np.sum(np.log(likelihoods)*test_vector,axis=1)
        # TODO Then get your predictions, yhat
        yhat = np.argmax(preds)
        class_predictions.append(yhat)
    return class_predictions


def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn implementation
    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """
    vectorizer = CountVectorizer()

    # TODO Fit the vectorizer on the training set text
    vectorizer.fit(training_df['text'])

    # TODO Then transform the text using the vectorizer
    training_data = vectorizer.transform(training_df['text'])
    training_data = training_data.toarray()

    # Do the same for the test data
    test_data = vectorizer.transform(test_df['text'])
    test_data = test_data.toarray()

    nb_classifier = MultinomialNB()
    # TODO Fit the Naive Bayes classifier
    nb_classifier.fit(training_data, training_df['author'])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb


def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    # TODO Compute performance measures
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds, average='weighted')
    conf_matrix = metrics.confusion_matrix(true, preds)
    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to make corpus " "for Naive Bayes homework"
    )
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)
    vocabulary, priors, likelihoods = train_nb(training_df)
    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    acc, f1, conf = get_metrics(test_df['author'], class_predictions)
    plot_confusion_matrix(conf, [0, 1])
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_metrics = get_metrics(test_df['author'], sklearn_preds)
    plot_confusion_matrix(sklearn_metrics[2],[0,1])
    
