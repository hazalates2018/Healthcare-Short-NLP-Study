import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import Word, TextBlob
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

df = pd.read_excel("C:\\Users\\hazal\\Downloads\\Employee_Performance_Evaluation_Form.xlsx")
df.head(20)
df.columns

# Normalizing Case Folding
df["Employee Performance Evaluation Form"] = df["Employee Performance Evaluation Form"].str.lower()
# Punctuations
df["Employee Performance Evaluation Form"] = df["Employee Performance Evaluation Form"].str.replace("[^\w\s]", "")
# Numbers
df["Employee Performance Evaluation Form"] = df["Employee Performance Evaluation Form"].str.replace("\d","")
# Stopwords
import nltk
 nltk.download("stopwords")
sw = stopwords.words("english")
df["Employee Performance Evaluation Form"] = df["Employee Performance Evaluation Form"].apply(lambda x: " ".join([x for x in str(x).split() if x not in sw]))

nltk.download("wordnet")
df["Employee Performance Evaluation Form"] = df["Employee Performance Evaluation Form"].apply(lambda x: "".join([Word(word).lemmatize() for word in x.split()]))


# barplot görselleştirme
tf = df["Employee Performance Evaluation Form"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words","tf"]
tf["tf"].plot.bar(x="words",y="tf")
plt.show()


# Wordcloud Görselleştirme

text = " ".join(i for i in df["Employee Performance Evaluation Form"])

wordcloud = WordCloud(max_font_size=50,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

