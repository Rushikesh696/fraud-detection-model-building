from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, countDistinct, isnan, count
from pyspark.sql.types import NumericType

spark = SparkSession.builder \
    .appName('Fraud Detection') \
    .getOrCreate()

df = spark.read.csv('fraud_detection_data.csv', header=True, inferSchema=True)

print('feature info')
df.printSchema()

df.show(10)
print(f'Total rows : {df.count()}')

# summary statistics
print('summary statistics')
df.describe().show()

#check class balance
print('fruad class balance')
df.groupby('fraud').count().show()

# transaction by type
print('transaction type count')
df.groupby('transaction_type').count().show()


numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
all_cols = df.columns

# Null values (for all columns)
print('Null value count:')
df.select([count(when(col(c).isNull(), c)).alias(c) for c in all_cols]).show()

# NaN values (only for numeric columns)
print('NaN value count (numeric columns only):')
df.select([count(when(isnan(col(c)), c)).alias(c) for c in numeric_cols]).show()


# check for duplicates
print('duplicate rows')
unique_count = df.dropDuplicates().count()
duplicate_count = df.count() - unique_count
print(f'Duplicate rows: {duplicate_count}')

# count unique values per column
print('unique value count per column')
df.select([countDistinct(c).alias(c) for c in df.columns]).show()

spark.stop()