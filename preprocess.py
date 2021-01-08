'''
algorithm to pull data from reddit,
determine whether the poster is an asshole based on the commenters annotattions,
and output to file
(maybe link with EC2?)
'''
import praw
import os

from praw.models import MoreComments

import config
from pathlib import Path
import csv

def does_commenter_think_OP_is_an_ass(comment_text: str) -> bool:
    """
    given the text from a comment, determine if the commenter thinks OP is an asshole or not
    :param comment_text: the commenter's text
    :return: True if the commenter thinks OP is an asshole
    """
    if 'YTA' in comment_text:
        return True
    if 'NTA' in comment_text:
        return False
    return None

def is_asshole(submission: praw.models.reddit.submission.Submission):
    """
    iterate through the comments to find signals for asshole or not
    NTA = not the asshole
    YTA = you're the asshole
    :param submission: a praw.models.reddit.submission.Submission object
    :return: True if more comments have 'YTA' than 'NTA'
    """
    votes_for_NTA = 0
    votes_for_YTA = 0

    # This submission’s comment forest contains a number of MoreComments objects.
    # These objects represent the “load more comments”,
    # and “continue this thread” links encountered on the website.
    # To remove them, we can call replace_more with a limit of 0
    # to remove ALL MoreComments
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        text = comment.body
        commenter_thinks_OP_is_ass = does_commenter_think_OP_is_an_ass(text)
        if commenter_thinks_OP_is_ass is None:
            continue
        if commenter_thinks_OP_is_ass:
            votes_for_YTA += 1
        else:
            votes_for_NTA += 1

    return votes_for_YTA > votes_for_NTA


if __name__ == '__main__':
    reddit = praw.Reddit(client_id=config.REDDIT_CLIENT_ID,
                         client_secret=config.REDDIT_CLIENT_SECRET,
                         user_agent=config.REDDIT_USER_AGENT,
                         username=config.REDDIT_USERNAME,
                         )

    print('connection successful')

    data_path = Path('./data/raw/debug.csv')
    num_assholes = 0
    with data_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation', 'title', 'text'])
        for i, submission in enumerate(reddit.subreddit("AmITheAsshole").new(limit=25)):
            title = submission.title
            text = submission.selftext
            annotation = is_asshole(submission)
            if annotation:
                num_assholes += 1
            writer.writerow([annotation, title, text])
    print(f'num assholes: {num_assholes} out of {i} total posts. Percent {num_assholes / i}')