def load_from_lsf(script_file_name):
    '''
    A very simple function that can load an lsf file and reformats it to work with this code.
    Effectively it strips out all comments from the file, because the comments don't play well with lumapi

    :param script_file_name: Just the name of the script
    :return:
    '''

    with open(script_file_name, 'rb') as text_file:
        lines = [line.strip().split('#',1)[0] for line in text_file.readlines()]

    script=""
    for line in lines:
        script+=line
    return script
