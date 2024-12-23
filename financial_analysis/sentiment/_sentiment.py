from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nltk.download("all")
analyzer = SentimentIntensityAnalyzer()


def sentiment_analysis(text_file: str) -> pd.DataFrame:
    """
    Performs sentiment analysis on a given text file.

    Parameters
        text_file: str
            The path to a text file containing text data to analyze.

    Returns
        df: pd.DataFrame
            A DataFrame containing the original text, preprocessed text, and sentiment scores
            (negative, neutral, positive, and compound).
    """

    # Preprocess text
    def preprocess_text(text):
        """
        Cleans and preprocesses text by tokenizing, removing stopwords, and lemmatizing.

        Parameters
            text: str
                The text to preprocess.

        Returns
            processed_text: str
                The preprocessed text.
        """
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            token for token in tokens if token not in stopwords.words("english")
        ]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = " ".join(lemmatized_tokens)
        return processed_text

    # Get sentiment scores
    def get_sentiment(text):
        """
        Calculates sentiment scores for the given text using a sentiment analyzer.

        Parameters
            text: str
                The text to analyze.

        Returns
            scores: dict
                A dictionary containing sentiment scores (negative, neutral, positive, and compound).
        """
        scores = analyzer.polarity_scores(text)
        return scores

    # Read text file
    with open(text_file, "r") as f:
        lines = [line.strip() for line in f]
    df = pd.DataFrame(lines, columns=["text"])

    # Preprocess text
    df["text"] = df["text"].apply(preprocess_text)

    # Get sentiment scores
    sentiment = pd.DataFrame(list(df["text"].apply(get_sentiment)))
    df = pd.concat([df, sentiment], axis=1)
    return df


def plot_sentiment(df: pd.DataFrame) -> None:
    """
    Visualizes sentiment scores as a stacked bar chart.

    Parameters
        df: pd.DataFrame
            A DataFrame containing sentiment scores ('neg', 'neu', 'pos') for each line of text.

    Returns
        None
            Displays a bar plot of sentiment scores.
    """
    df_sentiment = df.copy().reset_index()

    sns.set_theme(style="darkgrid", palette="muted")
    plt.style.use("dark_background")
    plt.figure(figsize=(20, 6))

    # Plot sentiment scores as stacked bars
    plt.bar(
        df_sentiment["index"], df_sentiment["neg"], label="Negative", color="#00246B"
    )
    plt.bar(
        df_sentiment["index"],
        df_sentiment["neu"],
        bottom=df_sentiment["neg"],
        label="Neutral",
        color="#8AB6F9",
    )
    plt.bar(
        df_sentiment["index"],
        df_sentiment["pos"],
        bottom=df_sentiment["neg"] + df["neu"],
        label="Positive",
        color="#CADCFC",
    )

    # Set plot labels
    plt.title("Sentiment Score Bar Plot", fontsize="15", color="white", pad=15)
    plt.ylabel("Sentiment Score", fontsize="13", color="white", labelpad=15)
    plt.yticks(fontsize=10)
    plt.xlabel("Lines", fontsize="13", color="white", labelpad=15)
    plt.xlim(-2, 151)
    plt.xticks(fontsize=10)

    # Set legend
    plt.legend(
        title="Sentiment",
        title_fontsize=13,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=13,
    )
    # Set grid
    plt.grid(True, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot
