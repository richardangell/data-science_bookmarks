import bs4

from bookmark import Bookmark



def parse_bookmarks_to_md(filename):
    """Function to do parse html bookmarks exported from Firefox to markdown file."""

    soup = parse_html_bookmarks(filename)

    dl = get_bookmarks_list(soup)

    bookmarks_dict = extract_dl_recursive(dl)

    data_science_bookmarks = get_data_science_bookmarks(bookmarks_dict)

    export_bookmark_icons(data_science_bookmarks, ['Data Science'])

    write_markdown_file(data_science_bookmarks, "#")

    return data_science_bookmarks



def parse_html_bookmarks(filename):

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
    return items



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



def export_bookmark_icons(bookmarks, folder_level):
    """Function to traverse the a nested structure containing Bookmark objects 
    and run the 
    """ 

    folder_level_depth = "__".join(folder_level)

    j = 0

    for k, v in bookmarks.items():

        for i, item in enumerate(v):

            if type(item) is dict:
                
                next_level = list(item.keys())[0]

                next_folder_level = folder_level + [next_level]

                export_bookmark_icons(item, next_folder_level)

            elif type(item) is Bookmark:
                
                if not item.icon is None:

                    item.export_icon_to_file(f'assets/images/{folder_level_depth}_{j}.png')
                    
                    j += 1

            else:

                raise TypeError(f'unexpected type ({type(item)}) at {folder_level_depth} and index {i}')



def write_markdown_file(bookmarks, folder_level):

    markdown_string = """---\ntitle: "Bookmarks"\npermalink: /bookmarks/\nsidebar:\n  title: "Navigation"\n  nav: bookmarks-sidebar\n---\n\n"""

    markdown_text = write_markdown_file_recursive(bookmarks, folder_level, markdown_string)

    markdown_text = markdown_text.replace('assets/images/', '/assets/images/')

    with open("_pages/bookmarks.md", "w") as index_md:

        index_md.write(markdown_text)



def write_markdown_file_recursive(bookmarks, folder_level, markdown_string):
    """Function to write index.md file containing bookmark info passed in dict."""

    for k, v in bookmarks.items():

        markdown_string = f'''{markdown_string}{folder_level} {k}\n\n'''

        for i, item in enumerate(v):
            
            if type(item) is dict:
                
                markdown_string = write_markdown_file_recursive(item, folder_level + "#", markdown_string)

            elif type(item) is Bookmark:
                
                if not item.icon_filename is None:

                    icon_text = f'''<img src="{item.icon_filename}" width="20" height="20"> '''

                else:

                    icon_text = ""

                bookmark_text = f'{icon_text}[{item.name}]({item.url})\n{item.url_netloc}\n\n'
                
                markdown_string = markdown_string + bookmark_text
                
    return markdown_string
   


if __name__ == '__main__':

    bookmarks_export = '/Users/richardangell/Desktop/bookmarks.html'

    markdown_string = parse_bookmarks_to_md(bookmarks_export)
