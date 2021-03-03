'''
algorithm to pull data from reddit,
determine whether the poster is an asshole based on the comments,
and output to file
(maybe link with EC2?)
'''
import csv
import logging
from pathlib import Path

import praw
from praw.models import MoreComments

from src import config


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

    upvotes_for_NTA = 0
    upvotes_for_YTA = 0

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
            upvotes_for_YTA += comment.ups
        else:
            votes_for_NTA += 1
            upvotes_for_NTA += 1

    return int(votes_for_YTA > votes_for_NTA), {'YTA_votes': votes_for_YTA,
                                                'YTA_upvotes': upvotes_for_YTA,
                                                'NTA_votes': votes_for_NTA,
                                                'NTA_upvotes': upvotes_for_NTA,
                                                }


def is_post_asking_AITA(title, text):
    title = title.lower()
    text = text.lower()
    if 'meta' in title or 'update' in title:
        return False
    if text == '[removed]' or text == '[deleted]':
        return False
    return True


if __name__ == '__main__':
    logging.basicConfig(
        # filename='preprocess.log', filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # console_level=logging.DEBUG,
    )
    reddit = praw.Reddit(client_id=config.REDDIT_CLIENT_ID,
                         client_secret=config.REDDIT_CLIENT_SECRET,
                         user_agent=config.REDDIT_USER_AGENT,
                         username=config.REDDIT_USERNAME,
                         )
    logging.info('connection successful')

    num_posts = 14_000
    data_path = Path(f'./data/raw/top-{num_posts}.csv')
    with data_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|')
        writer.writerow(['is_asshole', 'YTA_votes', 'YTA_upvotes', 'NTA_votes', 'NTA_upvotes', 'post_upvotes',
                         'title', 'text'])

        total_submissions = 0
        num_assholes = 0

        # we'll to paginate based on time. praw took away the ability to paginate on id :(
        latest_fullname = None
        while total_submissions < num_posts:
            submissions = reddit.subreddit("AmITheAsshole").top(limit=100,
                                                                params={'after': latest_fullname},
                                                                )
            for submission in submissions:
                latest_fullname = submission.fullname
                title = submission.title
                text = submission.selftext.replace('\n', ' ')
                if not is_post_asking_AITA(title, text):
                    # these aren't the posts we're looking for
                    continue

                annotation, info = is_asshole(submission)
                if annotation:
                    num_assholes += 1
                writer.writerow([annotation,
                                 info['YTA_votes'], info['YTA_upvotes'],
                                 info['NTA_votes'], info['NTA_upvotes'],
                                 submission.ups,
                                 title, text])
                total_submissions += 1
                if total_submissions % 100 == 0:
                    logging.info(f'{total_submissions} instances complete. {num_assholes} assholes found. '
                                 f'{num_assholes / total_submissions * 100}%')

    logging.info(f'num assholes: {num_assholes} out of {total_submissions} total posts. '
                 f'Percent {num_assholes / total_submissions}')
