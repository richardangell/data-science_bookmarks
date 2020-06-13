from pprint import pprint
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

    write_navigation_yml(data_science_bookmarks)

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
    return sort_bookmarks_list(items)



def sort_bookmarks_list(l):
    """Function to sort list containing dict and Bookmark objects.

    All Bookmark objects will appear before dict object. Apart from that
    there are not other conditions considered in the 'sorting'.
   """

    # l_sorted = l[:]

    # for i in range(len(l)-1, -1, -1):

    #     if type(l[i]) is dict:

    #         l_sorted.append(l[i])
    #         del l_sorted[i]

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
    """Function to write the _pages/bookmarks.md file, containing bookmark information."""

    markdown_string = "---\n"
    markdown_string = markdown_string + """title: "Bookmarks Export"\n"""
    markdown_string = markdown_string + "permalink: /bookmarks/\n"
    markdown_string = markdown_string + "sidebar:\n"
    markdown_string = markdown_string + """  title: "Navigation"\n"""
    markdown_string = markdown_string + "  nav: bookmarks-sidebar\n"
    markdown_string = markdown_string + "---\n"
    markdown_string = markdown_string + "\n"
    markdown_string = markdown_string + "This pages contains a export of my data science bookmarks folder. It is not updated on any sort of schedule.\n"
    markdown_string = markdown_string + "\n"

    markdown_text = write_markdown_file_recursive(bookmarks, folder_level, markdown_string)

    markdown_text = markdown_text.replace('assets/images/', '/data-science_bookmarks/assets/images/')

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

                bookmark_text = f'{icon_text}[{item.name}]({item.url})\n({item.url_netloc})\n\n'
                
                markdown_string = markdown_string + bookmark_text
                
    return markdown_string
   


def write_navigation_yml(bookmarks):
    """Function to write the _data/navigation.yml file according to the structure of the parsed bookmarks.
    
    Each top level folder (with the Data Science folder) will become a separate section in the 
    navigation side bar, with all subfolders within that folder listed beneath (without further
    nesting).
    """

    yml_string = """main:\n  - title: "Bookmarks"\n    url: /bookmarks/\n  - title: "About"\n    url: /about/\n\n"""

    yml_string = yml_string + """bookmarks-sidebar:\n"""
    
    folder_name = 'Data Science'
    yml_string = f"""{yml_string}  - title: "{folder_name}"\n    children:\n"""
    yml_string = yml_string + f"""      - title: "{folder_name}"\n"""
    yml_string = yml_string + f"""        url: /bookmarks/#{folder_name.replace(' ', '-').lower()}\n"""

    for i, item in enumerate(bookmarks['Data Science']):

        if type(item) is dict:

            folder_name = list(item.keys())[0]

            yml_string = f"""{yml_string}  - title: "{folder_name}"\n    children:\n"""
            yml_string = yml_string + f"""      - title: "{folder_name}"\n"""
            yml_string = yml_string + f"""        url: /bookmarks/#{folder_name.replace(' ', '-').lower()}\n"""


            yml_string = write_navifation_yml_recursive(item, yml_string)

    with open("_data/navigation.yml", "w") as navigation_yml:

        navigation_yml.write(yml_string)



def write_navifation_yml_recursive(bookmarks, yml_string):
    """Function to recursively transition down bookmarks dict and add subfolders to
    string that will be written to nagivation yml file.
    """

    for k, v in bookmarks.items():

        for i, item in enumerate(v):
            
            if type(item) is dict:
                
                folder_name = list(item.keys())[0]

                yml_string = yml_string + f"""      - title: "{folder_name}"\n"""
                yml_string = yml_string + f"""        url: /bookmarks/#{folder_name.replace(' ', '-').lower()}\n"""

                yml_string = write_navifation_yml_recursive(item, yml_string)

    return yml_string



if __name__ == '__main__':

    bookmarks_export = '/Users/richardangell/Desktop/bookmarks.html'

    bookmarks_dict = parse_bookmarks_to_md(bookmarks_export)

    pprint(bookmarks_dict)