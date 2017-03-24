import requests
from bs4 import BeautifulSoup
import codecs
import time
import math
# import ipdb


def get_movies(start_year, end_year, genre, sort='boxoffice_gross_us,desc', view='simple', pages=4):
    """

    """
    movies = []


    for page in range(pages):
        url = "http://www.imdb.com/search/title?year={start_year},{end_year}&title_type=feature&explore=genres".format(start_year=start_year, end_year=end_year)
        url += "&genres={genre}".format(genre=genre)
        url += "&view={view}".format(view=view)
        url += "&sort={sort}".format(sort=sort)
        url += "&page={page}".format(page=page+1)
        print 'url:', url
        html = requests.get(url).content
        soup = BeautifulSoup(html, 'html.parser')
        indexes = soup.select('.lister-list .lister-item-index')
        a_tags = soup.select('.lister-list .col-title a')

        for index, tag in zip(indexes, a_tags):
            index = index.text.replace(".", "")
            movie = tag.text

            href = tag['href']
            url_href = "http://www.imdb.com" + href
            movie_html = requests.get(url_href).content
            movie_soup = BeautifulSoup(movie_html, 'html.parser')

            movie_genres = movie_soup.select('.title_wrapper a span.itemprop')
            # print movie_genres

            genres = []
            for _genre in movie_genres:
                genres.append(_genre.text)

            genres = '"{}"'.format(','.join(genres))

            movies.append((index, movie, genres))

        print '...finished getting movies for page {} of {}'.format(page, genre)
        print 'elapsed time: {}'.format(time_elapsed(start_time))

    return movies

def get_imdb_genres():
    genres = []

    url = "http://www.imdb.com/search/title?title_type=feature&explore=genres"
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    genre_cells = soup.select('#sidebar .aux-content-widget-2 table td a')

    for genre_cell in genre_cells:
        genres.append(genre_cell.text)

    return genres

def write_movies_to_csv(movies, filename):
    print 'start writing files...'
    print 'elapsed time: {}'.format(time_elapsed(start_time))
    with codecs.open(filename,'wb', 'utf-8') as file:
        for movie in movies:
            file.write(','.join(movie))
            file.write('\n')
    print '...finished writing files'
    print 'elapsed time: {}'.format(time_elapsed(start_time))

def time_elapsed(start_time):
    elapsed_time = float(time.time()) - float(start_time)
    minutes = math.floor(elapsed_time / 60)
    seconds = elapsed_time % 60
    return "{} minutes and {} seconds".format(minutes, seconds)

if __name__ == '__main__':
    start_time = time.time()

    print 'getting genres...'
    genres = get_imdb_genres()
    print '...finished getting genres'

    for genre in genres:
        print 'getting movies for {}...'.format(genre)
        movies = get_movies(2011, 2017, genre)
        filename = 'data/{}.csv'.format(genre.lower())
        write_movies_to_csv(movies, filename)
        print '...finished getting all movies for {}'.format(genre)
        print 'elapsed time: {}'.format(time_elapsed(start_time))
