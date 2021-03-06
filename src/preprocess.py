'''
algorithm to pull data from reddit,
determine whether the poster is an asshole based on the comments,
and output to file
(maybe link with EC2?)
'''
import csv
import logging
from collections import Counter
from pathlib import Path

import praw
from praw.models import MoreComments

from src import config
from src.class_enum import ClassEnum



def does_commenter_think_OP_is_an_ass(comment_text: str):
    """
    given the text from a comment, determine if the commenter thinks OP is an asshole or not
    :param comment_text: the commenter's text
    :return: The enum corresponding to the type of comment
    """
    for type in ClassEnum:
        if type.name in comment_text:
            return type
    return None


def is_asshole(submission: praw.models.reddit.submission.Submission):
    """
    iterate through the comments to find signals for asshole or not
    see src\class_enum.py for possible types
    :param submission: a praw.models.reddit.submission.Submission object
    :return: True if more comments have 'YTA' than 'NTA'
    """
    # following the subreddit's rules, we take the top voted comment to be the annotation
    # see more [here](https://www.reddit.com/r/AmItheAsshole/wiki/faq#wiki_why_do_you_use_upvotes_to_determine_the_winning_judgment.3F_why_don.2019t_you_count_up_the_nta_and_yta_in_each_thread.3F)
    commenter_thinks_OP_is = None
    submission.comment_sort = 'top'
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        text = comment.body
        commenter_thinks_OP_is = does_commenter_think_OP_is_an_ass(text)
        if commenter_thinks_OP_is is None:
            # skip over moderator comments (default to top position)
            continue
        # we've found the top comment that labels OP as a class
        break

    return commenter_thinks_OP_is


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
    # api = PushshiftAPI(reddit)  # this allows for pagination of praw results
    logging.info('connection successful')

    num_posts = 850
    data_path = Path(f'./data/raw/top-{num_posts}.csv')
    with data_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|')
        writer.writerow(['fullname', 'is_asshole',
                         'post_upvotes',
                         'title', 'text'])

        total_submissions = 0
        annotations = []

        # we'll to paginate based on time. praw took away the ability to paginate on id :(
        latest_fullname = None
        while total_submissions < num_posts:
            submissions = reddit.subreddit("AmITheAsshole").top(limit=100,
                                                                params={'after': latest_fullname,
                                                                        'comment_sort': 'top'},
                                                                )
            # end_epoch = start_epoch -  datetime.timedelta(days=10)
            # submissions = api.search_submissions(limit=None,
            #                                      before=int(start_epoch.timestamp()),
            #                                      after=int(end_epoch.timestamp()),
            #                                      subreddit='AmITheAsshole',
            #                                      )
            # start_epoch = end_epoch

            for submission in submissions:
                title = submission.title
                text = submission.selftext.replace('\n', ' ')
                if not is_post_asking_AITA(title, text):
                    # these aren't the posts we're looking for
                    continue

                latest_fullname = submission.fullname
                annotation = is_asshole(submission)
                annotations.append(annotation.name)
                writer.writerow([latest_fullname, annotation.name,
                                 submission.ups,
                                 title, text])
                total_submissions += 1
                if total_submissions % 100 == 0:
                    c = Counter(annotations)
                    percentage_dict = {key: val / total_submissions for key, val in c.items()}
                    logging.info(f'{total_submissions} instances complete. \n'
                                 f'{c}\n'
                                 f'        '
                                 f'{percentage_dict}')
    c = Counter(annotations)
    percentage_dict = {key: val / total_submissions for key, val in c.items()}
    logging.info(f'{total_submissions} total posts. '
                 f'{c}\n'
                 f'        '
                 f'{percentage_dict}'
                 )
