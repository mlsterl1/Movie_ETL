# Movie_ETL
* Use wikipdeia.movies.json, kaggle_data (movies_metadata.csv) and ratings.csv to extract, transform and load data into sql tables.
* Inspect , clean and organize data for accuracy 
* Create one function  to import, clean and transform data. 
* Create a movie and ratings table in the sql 
## Assumptions 
* Box office/Budget data has two main forms : 1234.million and 123,456,789. 
* Release date has four forms: 
  * Full month name, one/two digit day and four digit year 
  * Four digit year, two digit month , two digit day and any seperator 
  * Four digit year
  * Full month name, four digit year
* Running time is convereted to hours , hours and minutes and minutes but not seconds. 
* Kaggle data is more structured / budget, id and popularity are all numeric. 
* There are common columns between kaggle_data and wikipedia. Kaggle_data seems to be more consistent and better than wikipedia. Kaggle dat is kept and wikipedia is dropped. 
* Ratings file is large and takes a long time to import to SQL database. 
* Challenge file removes alot of the explortory code you would need in order to understand and clean data. 
