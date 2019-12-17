import pandas as pd
reviews = pd.read_csv("ign.csv")
print(reviews)

reviews.shape
reviews.head()
#Earlier, we used the head method to print the first 5 rows of reviews. We could accomplish the same thing using the pandas.DataFrame.iloc method. The iloc method allows us to retrieve rows and columns by position. In order to do that, we’ll need to specify the positions of the rows that we want, and the positions of the columns that we want as well. The below code will replicate the results of our reviews.head() by selecting rows zero to five, and all of the columns in our data set:
reviews.iloc[0:5,:]
#
#reviews.iloc[:5,:] — the first 5 rows, and all of the columns for those rows.
#reviews.iloc[:,:] — the entire DataFrame.
#reviews.iloc[5:,5:] — rows from position 5 onwards, and columns from position 5 onwards.
#reviews.iloc[:,0] — the first column, and all of the rows for the column.
#reviews.iloc[9,:] — the 10th row, and all of the columns for that row.

reviews.index
some_reviews = reviews.iloc[10:20,]
some_reviews.head()
some_reviews.loc[9:21,:]
reviews.loc[:5,"score"]
reviews.loc[:5,["score", "release_year"]]

#Pandas Series Objects
#=====================
#We can retrieve an individual column in Pandas a few different ways. So far, we’ve seen two types of syntax for this:

	#reviews.iloc[:,1] — will retrieve the second column.
	#reviews.loc[:,"score_phrase"] — will also retrieve the second column.
#There’s a third, even easier, way to retrieve a whole column. We can just specify the column name in square brackets, like with a dictionary:

reviews["score"]

reviews[["score", "release_year"]]

type(reviews["score"])

s1 = pd.Series([1,2])
s1

s2 = pd.Series(["Boris Yeltsin", "Mikhail Gorbachev"])
s2

#Creating A DataFrame in Pandas
#===============================
pd.DataFrame([s1,s2])

#We can also accomplish the same thing with a list of lists. Each inner list is treated as a row in the resulting DataFrame:

pd.DataFrame(
    [
    [1,2],
    ["Boris Yeltsin", "Mikhail Gorbachev"]
    ]
)

#We can specify the column labels when we create a DataFrame:

pd.DataFrame(
    [
    [1,2],
    ["Boris Yeltsin", "Mikhail Gorbachev"]
    ],
    columns=["column1", "column2"]
)

frame = pd.DataFrame([[1,2],["Boris Yeltsin", "Mikhail Gorbachev"]],index=["row1", "row2"],columns=["column1", "column2"])

frame.loc["row1":"row2", "column1"]

#We can skip specifying the columns keyword argument if we pass a dictionary into the DataFrame constructor. This will automatically set up column names:

frame = pd.DataFrame(
    {
    "column1": [1, "Boris Yeltsin"],
    "column2": [2, "Mikhail Gorbachev"]
    }
)
frame

#Pandas DataFrame Methods
#========================
type(reviews["title"])
reviews["title"].head()
reviews["score"].mean()
reviews.mean()
reviews.mean(axis=1)

reviews.corr()

#pandas.DataFrame.corr — finds the correlation between columns in a DataFrame.
#pandas.DataFrame.count — counts the number of non-null values in each DataFrame column.
#pandas.DataFrame.max — finds the highest value in each column.
#pandas.DataFrame.min — finds the lowest value in each column.
#pandas.DataFrame.median — finds the median of each column.
#pandas.DataFrame.std — finds the standard deviation of each column.

#DataFrame Math with Pandas
#==========================
reviews["score"] / 2

#Boolean Indexing in Pandas
score_filter = reviews["score"] > 7
score_filter

filtered_reviews = reviews[score_filter]
filtered_reviews.head()

xbox_one_filter = (reviews["score"] > 7) & (reviews["platform"] == "Xbox One")
filtered_reviews = reviews[xbox_one_filter]
filtered_reviews.head()

#Pandas Plotting
#================
reviews[reviews["platform"] == "Xbox One"]["score"].plot(kind="hist")

reviews[reviews["platform"] == "PlayStation 4"]["score"].plot(kind="hist")