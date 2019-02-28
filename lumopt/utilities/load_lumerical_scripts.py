""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

def load_from_lsf(script_file_name):
    """ 
       Loads the provided scritp as a string and strips out all comments. 

       Parameters
       ----------
       :param script_file_name: string specifying a file name.
    """

    with open(script_file_name, 'r') as text_file:
        lines = [line.strip().split(sep = '#', maxsplit = 1)[0] for line in text_file.readlines()]
    script = ' '.join(lines)
    if not script:
        raise UserWarning('empty script.')
    return script
