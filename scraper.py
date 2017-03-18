def get_movies(year):
    """

    """
    url = "http://www.indeed.com/jobs?q=%s" % query.replace(' ', '+')
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.select('#searchCount')[0].text
    keyword = 'of '
    befor_keyowrd, keyword, after_keyword = text.partition(keyword)
    return after_keyword
