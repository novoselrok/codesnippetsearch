from pygments.formatters import HtmlFormatter


def get_pygments_html_formatter():
    return HtmlFormatter(linenos=False, style='xcode')
