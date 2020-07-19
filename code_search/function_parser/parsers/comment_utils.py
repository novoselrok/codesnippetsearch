def strip_c_style_comment_delimiters(comment: str) -> str:
    comment_lines = comment.split('\n')
    cleaned_lines = []
    for line in comment_lines:
        line = line.strip()
        if line.endswith('*/'):
            line = line[:-2]
        if line.startswith('*'):
            line = line[1:]
        elif line.startswith('/**'):
            line = line[3:]
        elif line.startswith('//') or line.startswith('/*'):
            line = line[2:]

        if len(line) > 0:
            cleaned_lines.append(line.strip())
    return '\n'.join(cleaned_lines)


def get_docstring_summary(docstring: str) -> str:
    """Get the first lines of the documentation comment up to the empty lines."""
    if '\n\n' in docstring:
        return docstring.split('\n\n')[0]
    elif '@' in docstring:
        return docstring[:docstring.find('@')]  # This usually is the start of a JavaDoc-style @param comment.
    return docstring
