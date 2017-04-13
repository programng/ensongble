# ensongble

Machine learning project to predict the genre of a movie using the movie's soundtrack. Stemmed from an interest in the added storytelling power of music and the scoring of movies.

#### process_data.py

* loads audio buffers from audio files
* serializes audio buffers by movie
* serializes features
* serializes final dataframe

#### generate_production_model.py

* generates the production model to deploy using dataframe created by process_data script

#### train_test_by_song.py

* train and test predictions of songs' genres

#### train_test_ensemble.py

* train and test using an ensemble approach for the predictions of movies' genres

#### scraper.py

* gets the top 200 movies' names and meta data from each movie genre from IMDB
