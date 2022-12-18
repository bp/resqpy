"""Basic functions for searching for keywords in an ascii control file such as a nexus deck.

Ascii file must already have been opened for reading before calling any of these functions.
"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)


def substring(shorter, longer):
    """Returns True if the first argument is a substring of the second."""
    try:
        longer.index(shorter)
        return True
    except Exception:
        return False


def end_of_file(ascii_file):
    """Returns True if the end of the file has been reached."""
    file_pos = ascii_file.tell()
    line = ascii_file.readline()
    if len(line) == 0:
        return True  # end of file
    ascii_file.seek(file_pos)
    return False


def find_keyword(ascii_file, keyword, max_lines = None):
    """Looks for line starting with given keyword; file pointer is left at start of that line."""
    start_pos = ascii_file.tell()
    while True:
        if max_lines is not None:
            if max_lines <= 0:
                ascii_file.seek(start_pos)
                return False
            max_lines -= 1
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            ascii_file.seek(start_pos)
            return False  # end of file
        words = line.split()
        if len(words) > 0 and words[0].upper() == keyword.upper():
            ascii_file.seek(file_pos)
            return True


def skip_blank_lines_and_comments(ascii_file, comment_char = '!', skip_c_space = True):
    """Skips any lines containing only white space or comment."""
    while True:
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            return  # end of file
        words = line.split()
        if len(words) == 0:
            continue
        if words[0][0] == comment_char:
            continue
        if skip_c_space and len(words[0]) == 1 and (words[0][0] == 'C' or words[0][0] == 'c'):
            continue
        ascii_file.seek(file_pos)
        return


def skip_comments(ascii_file, comment_char = '!', skip_c_space = True):
    """Skips any lines containing only a comment."""
    while True:
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            return  # end of file
        words = line.split()
        if len(words) > 0:  # not a blank line
            if words[0][0] == comment_char:
                continue
            if skip_c_space and len(words[0]) == 1 and (words[0][0] == 'C' or words[0][0] == 'c'):
                continue
        ascii_file.seek(file_pos)
        return


def split_trailing_comment(line, comment_char = '!'):
    """Returns a pair of strings: (line stripped of trailing comment, trailing comment)."""
    # also removes trailing newline
    comment = ''
    local_line = line
    while len(local_line) > 0 and local_line[-1] in ['\n', '\r']:
        local_line = local_line[:-1]
    try:
        pling = local_line.index(comment_char)
        if pling < len(local_line) - 2:
            comment = local_line[pling + 1:]
        return (local_line[:pling], comment)
    except Exception:
        return (local_line, '')


def strip_trailing_comment(line, comment_char = '!'):
    """Returns a copy of line with any trailing comment removed."""
    (result, comment) = split_trailing_comment(line, comment_char = comment_char)
    return result


def find_keyword_without_passing(ascii_file, keyword, no_pass_keyword):
    """Looks for line starting with keyword, but without passing line starting with no_pass_keyword."""
    while True:
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            return False  # end of file
        words = line.split()
        if len(words) > 0:
            if words[0].upper() == keyword.upper():
                ascii_file.seek(file_pos)
                return True
            if words[0].upper() == no_pass_keyword.upper():
                ascii_file.seek(file_pos)
                return False


def find_keyword_with_copy(ascii_file_in, keyword, ascii_file_out):
    """Looks for line starting with given keyword, copying lines in the meantime."""
    while True:
        file_pos = ascii_file_in.tell()
        line = ascii_file_in.readline()
        if len(line) == 0:
            return False  # end of file
        words = line.split()
        if len(words) > 0 and words[0].upper() == keyword.upper():
            ascii_file_in.seek(file_pos)
            return True
        else:
            ascii_file_out.write(line)


def find_keyword_pair(ascii_file, primary_keyword, secondary_keyword):
    """Looks for line starting with a given pair of keywords."""
    while True:
        if not find_keyword(ascii_file, primary_keyword):
            return False
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            return False  # end of file
        words = line.split()
        if len(words) >= 2 and words[1].upper() == secondary_keyword.upper():
            ascii_file.seek(file_pos)
            return True


def find_number(ascii_file):
    """Looks for line starting with any number."""
    while True:
        file_pos = ascii_file.tell()
        line = ascii_file.readline()
        if len(line) == 0:
            return False  # end of file
        words = line.split()
        if len(words) > 0:
            try:
                float(words[0])
            except Exception:  # todo: should only catch a particular exception (type conversion)
                continue
            else:
                ascii_file.seek(file_pos)
                return True


def specific_keyword_next(ascii_file, keyword, skip_blank_lines = True, comment_char = '!'):
    """Returns True if next token in file is the specified keyword."""
    file_pos = ascii_file.tell()  # will restore file pos to original, which may be before blank lines
    while True:
        if (skip_blank_lines):
            file_pos = ascii_file.tell()  # will advance file pos to next non-blank line
        line = ascii_file.readline()
        if len(line) == 0:  # end of file
            ascii_file.seek(file_pos)
            return False
        words = line.split()
        if len(words) == 0 or words[0][0] == comment_char:
            continue  # blank line or comment
        ascii_file.seek(file_pos)  # restore file position whether or not keyword match
        if words[0].upper() == keyword.upper():
            return True
        return False


def number_next(ascii_file, skip_blank_lines = True, comment_char = '!'):
    """Returns True if next token in file is a number."""
    file_pos = ascii_file.tell()
    while True:
        if (skip_blank_lines):
            file_pos = ascii_file.tell()  # will advance file pos to next non-blank line
        line = ascii_file.readline()
        if len(line) == 0:  # end of file
            ascii_file.seek(file_pos)
            return False
        words = line.split()
        if len(words) == 0 or words[0][0] == comment_char:
            continue  # blank line or comment
        ascii_file.seek(file_pos)  # restore file position whether or not number found
        try:
            float(words[0])
        except Exception:  # todo: should only catch a particular exception (type conversion)
            return False
        else:
            return True


def blank_line(ascii_file):
    """Returns True if the next line contains only white space; False otherwise (including comments)."""

    file_pos = ascii_file.tell()
    line = ascii_file.readline()
    ascii_file.seek(file_pos)
    if len(line) == 0:
        return True  # end of file
    words = line.split()
    return len(words) == 0


def guess_comment_char(ascii_file):
    """Returns a string (usually one character) being the guess as to the comment character, or None."""
    file_pos = ascii_file.tell()  # will restore file pos to original
    max_lines = 10
    ch = None
    while max_lines:
        line = ascii_file.readline()
        max_lines -= 1
        if not line:
            break
        words = line.split()
        if len(words) and words[0] in ['!', 'C', '#', '--']:
            ch = words[0]
            break
    ascii_file.seek(file_pos)
    return ch


# end of keyword_files module
