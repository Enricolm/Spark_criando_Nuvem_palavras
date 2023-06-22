# %%
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import IntegerType
import nltk
from pyspark.ml.feature import StopWordsRemover
nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.ml.feature import IDF
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
# %%
spark = SparkSession.builder \
    .master('local[*]')\
    .appName("NLP")\
    .getOrCreate()

# %%
dados = spark.read.csv("/home/enricolm/Documents/SparkNLP/imdb-reviews-pt-br.csv",
    escape = '\"',
    inferSchema = True,
    header = True                       
)

# %%
dados = dados.withColumn('sentiment', f.when(f.col('sentiment') == "neg", "NEGATIVO").otherwise('POSITIVO'))


# %%
dados = dados.withColumn('texto_regex_pt',f.regexp_replace('text_pt',"[\$#,\"!%&'()*+-./;;<=>?@^_`´{|}~\\\\]",""))
dados = dados.withColumn('texto_regex_en',f.regexp_replace('text_en',"[\$#,\"!%&'()*+-./;;<=>?@^_`´{|}~\\\\]",""))

# %%
dados = dados.withColumn('texto_regex_pt', f.trim(dados['texto_regex_pt']))
dados = dados.withColumn('texto_regex_en', f.trim(dados['texto_regex_en']))


# %%

stop_pt = stopwords.words('portuguese')


# %%

stop_en = StopWordsRemover.loadDefaultStopWords('english')


# tf = HashingTF(inputCol= 'texto_StopWords_pt', outputCol='texto_TF_pt', numFeatures= 100)

# tf = HashingTF(inputCol= 'texto_StopWords_en', outputCol='texto_TF_en', numFeatures= 100)

# idf = IDF(inputCol='texto_TF_pt', outputCol='texto_IDF_pt')

# idf = IDF(inputCol= 'texto_TF_en', outputCol='texto_IDF_en')


si = StringIndexer(inputCol='sentiment' , outputCol= 'label')

dados = si.fit(dados).transform(dados)

dados.na.drop()


tokenizer = Tokenizer(inputCol='texto_regex_pt' , outputCol='texto_tokenizado_pt')
tokenizer_en = Tokenizer(inputCol= 'texto_regex_en', outputCol= 'texto_tokenizer_en')
remover = StopWordsRemover(inputCol='texto_tokenizado_pt', outputCol= 'texto_StopWords_pt', stopWords= stop_pt)
remover_en = StopWordsRemover(inputCol='texto_tokenizer_en', outputCol= 'texto_StopWords_en', stopWords= stop_en)
tf = HashingTF(inputCol= remover.getOutputCol(), outputCol='texto_TF_pt', numFeatures= 1000)
tf_en = HashingTF(inputCol= remover_en.getOutputCol(), outputCol='texto_TF_en', numFeatures= 1000)
idf = IDF(inputCol='texto_TF_pt', outputCol='texto_IDF_pt')
idf_en = IDF(inputCol= 'texto_TF_en', outputCol='texto_IDF_en')


pipeline = Pipeline(stages=[tokenizer,tokenizer_en,remover,remover_en,tf,tf_en,idf,idf_en])

# %%
dados_pipe = pipeline.fit(dados).transform(dados)

# %%
dados_pipe.show(truncate=False)