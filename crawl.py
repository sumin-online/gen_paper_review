import argparse
from glob import glob
from pathlib import Path
import pickle as pkl
import random
from time import sleep
from typing import Any, Dict, List

import openreview
from tqdm import tqdm

def get_review_content(review: dict, weakness: bool) -> Dict[str, str]:
    res = {}
    if weakness:
        try:
            res["weaknesses"] = review["content"]["weaknesses"]
            res["strengths"] = review["content"]["strengths"]
            res["review"] = review["content"]["summary and contributions"]
            res["rating"] = review["content"]["rating"]
        except KeyError:
            pass

    else:
        try:
            res["review"] = review["content"]["review"]
            res["rating"] = review["content"]["rating"]
        except KeyError:
            pass

    return res

def get_reviews(venue: str, submissions: List[openreview.Note], weakness: bool) -> List[Dict[str, str]]:
    result = []
    for submission in submissions:
        content: str = submission.content
        _id: str = content.get("id")
        tl_dr: str = content.get("TL;DR")
        abstract: str = content.get("abstract")
        title: str = content.get("title")
        reviews: List[dict] = [reply for reply in submission.details["directReplies"]
                   if reply["invitation"].endswith("Official_Review")]
        decision: List[dict] = [reply for reply in submission.details["directReplies"]
                    if reply["invitation"].endswith("Decision")]
        if len(decision) != 1:
            continue

        decision: str = decision[0]["content"]["decision"]
        accepted: bool = True if "Accept" in decision else False
        for review in reviews:
            temp = get_review_content(review, weakness)
            if len(temp.keys()) != 0:
                temp["id"] = _id
                temp["venue"] = venue
                temp["tl_dr"] = tl_dr
                temp["abstract"] = abstract
                temp["title"] = title
                temp["accepted"] = accepted
                result.append(temp)
    return result

def main(weakness: bool) -> None:
    client = openreview.Client(baseurl="https://api.openreview.net")
    venues: List[str] = client.get_group(id="venues").members

    result = []
    for venue in tqdm(venues):
        subs_blind: List[openreview.Note] = client.get_all_notes(invitation=venue+"/-/Blind_Submission",
                                          details="directReplies")
        subs_single: List[openreview.Note] = client.get_all_notes(invitation=venue+"/-/Submission",
                                          details="directReplies")
        submissions: List[openreview.Note] = subs_blind + subs_single
        reviews: List[dict[str, str]] = get_reviews(venue, submissions, args.weaknesses)
        result.extend(reviews)

    filename = ("reviews_with_weaknesses.pkl" if args.weaknesses
                else "reviews_without_weaknesses.pkl")

    pkl.dump(result, open(f"crawled/{filename}", "wb"))

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weaknesses", "-w", 
                        type=str_to_bool, 
                        nargs="?",
                        const=True, 
                        default=True, 
                        help="Whether crawl weaknesses or not")
    args = parser.parse_args()
    main(args.weaknesses)
