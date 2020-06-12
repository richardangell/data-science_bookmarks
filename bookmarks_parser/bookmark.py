import base64



class Bookmark(object):
    """Class to contain (a single) bookmark, exported from Firefox."""

    def __init__(self, name, url, icon = None, icon_extra_text = 'data:image/png;base64,', icon_print_truncation = 50):
        """init method
        
        Parameters
        ----------
        name : str
            The name of the bookmark.

        url : str
            The url of the bookmark.
        
        icon : str, default = None
            Base 64 encoded image for the bookmark icon, stored a string.

        icon_extra_text : str, default = 'data:image/png;base64,'
            Extra text included in the icon string that needs to be removed.

        icon_print_truncation : int, default = 50
            Limit on the number of characters to be printed in icon.

        """

        self.name = name
        self.url = url
        self.icon_extra_text = icon_extra_text
        if icon is None:
            self.icon = icon
        else:
            self.icon = self.clean_icon_string(icon)
        self.icon_print_truncation = icon_print_truncation
        self.icon_filename = None


    def __str__(self):

        if self.icon is None:
            icon_for_print = 'None'
        else:
            icon_for_print = f'{self.icon[:self.icon_print_truncation]}... truncated to {self.icon_print_truncation} characters'

        return f'name: {self.name}\nurl: {self.url}\nicon: {icon_for_print}'

    def __reprb__(self):

        if self.icon is None:
            icon_for_print = 'None'
        else:
            icon_for_print = f'{self.icon[:self.icon_print_truncation]}... truncated to {self.icon_print_truncation} characters'

        return f'name: {self.name}\nurl: {self.url}\nicon: {icon_for_print}'


    def clean_icon_string(self, s):
        """Function to remove self.icon_extra_text from string.
        
        Raises
        ------
        ValueError
            If self.icon_extra_text is not contained within s.

        """

        if not self.icon_extra_text in s:

            raise ValueError(f"""expecting '{self.icon_extra_text}' to be in s but it's not, s (first 50 characters) = '{s[:50]}'""")

        else:

            return s.replace(self.icon_extra_text, '')


    def export_icon_to_file(self, filename = None):
        """Method to write base 64 encoded image string (self.icon) to file.
        
        The passed filename is set to the icon_filename attribute.

        Parameters
        ----------
        filename : str
            File to write image to.
        
        Raises
        ------
        ValueError
            If img is not a string.

        """

        if filename is None:

            return None

        if self.icon is None:

            raise ValueError('icon is None - cannot export')

        img_bytes = str.encode(self.icon)

        img_64_decode = base64.decodebytes(img_bytes) 

        with open(filename, 'wb') as f:
            
            f.write(img_64_decode)

            self.icon_filename = filename

