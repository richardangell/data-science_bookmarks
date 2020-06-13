---
permalink: /about/
title: "About"
---

This site contains my data science related bookmarks from Firefox. They have been built up over several years and I try to keep them curated with useful links rather than dumping as much as possible in there. There were 3 main reasons behind producing this site;
- to try out [Github pages](https://pages.github.com/)
- to make my bookmarks available when I'm not signed into Firefox
- to create a resource that could be interesting for others to dip into and find reading material

The code used to parse the bookmarks (once exported manually to a html file) is in [bookmarks_parser](https://github.com/richardangell/data-science_bookmarks/tree/master/bookmarks_parser). 
In summary;
1. the html structure is parsed
2. icons are saved to the `assets/images/` folder
3. the parsed structure is traversed and 
    1. bookmark info is written to the `_pages/bookmarks.md` file - which is the bookmarks page on the site
    2. the folder structure is written to the `_data/navigation.yml` file - this defines the sidebar on the bookmarks page

The site uses the [minimal mistakes](https://mademistakes.com/work/minimal-mistakes-jekyll-theme/) jekyll theme.
