from pprint import pprint
import parser 

from bookmark import Bookmark



def update_site_files(html_bookmark_export_file):
    """Function to do parse html bookmarks exported from Firefox to markdown file."""

    data_science_bookmarks = parser.parse_bookmarks_wrapper(html_bookmark_export_file)

    parser.export_bookmark_icons(data_science_bookmarks, ['Data Science'])

    write_markdown_file(data_science_bookmarks, "#")

    write_navigation_yml(data_science_bookmarks)

    return data_science_bookmarks



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
    markdown_string = markdown_string + "This pages contains a export of my data science bookmarks folder. It is not updated on any sort of schedule. It hopefully contains useful and interesting topics across a variety of areas within data science.\n"
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

    bookmarks_dict = update_site_files(bookmarks_export)

    pprint(bookmarks_dict)
