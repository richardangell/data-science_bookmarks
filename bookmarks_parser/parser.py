from pprint import pprint
import bs4

import exporter
from bookmark import Bookmark



def parse_bookmarks_wrapper(html_bookmark_export_file):
    """Function to do parse html bookmarks exported from Firefox to dict."""

    soup = load_html_bookmarks(html_bookmark_export_file)

    dl = get_bookmarks_list(soup)

    bookmarks_dict = extract_dl_recursive(dl)

    data_science_bookmarks = get_data_science_bookmarks(bookmarks_dict)

    return data_science_bookmarks



def load_html_bookmarks(filename):

    with open(filename, 'rb') as f:

        soup = bs4.BeautifulSoup(f, "html5lib")

    return soup



def get_bookmarks_list(html):
    """Function to return this first dl tag in the passed html.

    For bookmarks exported from Firefox to html.
    """

    first_dl_tag = html.find_all('dl')[0]

    return first_dl_tag



def extract_dl_recursive(dl):
    items = []
    for child in dl:
        if child.name == 'dt':
            item_data = extract_dt_data(child)
            if not type(item_data) is Bookmark:
                k = list(item_data.keys())[0]
                item_data[k] = extract_dl_recursive(item_data[k])
            items.append(item_data)    
    return sort_bookmarks_list(items)



def sort_bookmarks_list(l):
    """Function to sort list containing dict and Bookmark objects.

    All Bookmark objects will appear before dict object. Apart from that
    there are not other conditions considered in the 'sorting'.
   """

    l_bookmarks = []
    l_dicts = []

    for item in l:

        if type(item) is dict:

            l_dicts.append(item)

        else:

            l_bookmarks.append(item)

    l_sorted = l_bookmarks + l_dicts

    return l_sorted



def extract_dt_data(dt):
    """Function to extract data from dt tag.
    
    The html is structure such that a dt tag will either contain an a tag or a h3 tag and dl tag.
    """
    
    for child in dt:

        if child.name == 'a':

            temp_bookmark = {
                'name': child.text,
                'url': child.get('href'),
                'icon': child.get('icon')
            }
            
            data = Bookmark(**temp_bookmark)
            
        elif child.name == 'h3':

            data = {}
            header = child.text
            data[header] = []

        elif child.name == 'dl':

            data[header] = child

    return data



def get_data_science_bookmarks(bookmarks):
    """Function to select the top level data science folder of bookmarks.
    
    Returns the first element of bookmarks where 'Data Science' is one of the keys.
    """

    for item in bookmarks:

        if type(item) is dict:

            if 'Data Science' in item.keys():

                return item 



if __name__ == '__main__':

    bookmarks_dict = parse_bookmarks_wrapper(exporter.bookmarks_export_file)

    pprint(bookmarks_dict)