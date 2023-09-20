import argparse

class ArgHelpFormatter(argparse.HelpFormatter):
    '''
    Formatter adding default values to help texts.
    '''
    def __init__(self, prog):
        super().__init__(prog)

    def _get_help_string(self, action):
        text = action.help
        if  action.default is not None and \
            action.default != argparse.SUPPRESS and \
            'default:' not in text.lower():
            text += ' (default: {})'.format(action.default)
        return text