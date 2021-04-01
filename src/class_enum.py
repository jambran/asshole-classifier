from enum import Enum


class ClassEnum(Enum):
    """
    See this subreddit FAQ for more about these label types
    https://www.reddit.com/r/AmItheAsshole/wiki/faq#wiki_what.2019s_with_these_acronyms.3F_what_do_they_mean.3F
    """
    NTA = 0  # not the asshole
    YTA = 1  # you're the asshole
    ESH = 2  # everyone sucks here
    NAH = 3  # no assholes here
    INFO = 4  # not enough info to determine
