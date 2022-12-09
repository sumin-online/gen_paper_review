import argparse
import pickle as pkl
from typing import Any, Dict, List, Union

import openreview
from tqdm import tqdm


def get_review_content(review: Dict[str, Any], weakness: bool) -> Dict[str, Union[str, bool]]:
    """
    Extract content (Strength, Weakness, overall review, rating) from review.

    :param review: Raw review content
    :param weakness: Whether to include weakness

    :return: Dictionary of processed review data.
    """
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


def get_reviews(
    venue: str, submissions: List[openreview.Note], weakness: bool
) -> List[Dict[str, Union[str, bool]]]:
    """
    Collect reviews from given venue

    :param venue: Determine reviews to collect.
    :param submissions: List of submitted reviews
    :param weakness: Whether to include weakness or not

    :return: Dictionary of extracted & processed reviews.
    """
    result = []
    for submission in submissions:
        content: Dict[str, Any] = submission.content
        _id: str = content.get("id")
        tl_dr: str = content.get("TL;DR")
        abstract: str = content.get("abstract")
        title: str = content.get("title")
        reviews: List[Dict[str, Any]] = [
            reply
            for reply in submission.details["directReplies"]
            if reply["invitation"].endswith("Official_Review")
        ]
        decision_dict: List[Dict[str, Any]] = [
            reply
            for reply in submission.details["directReplies"]
            if reply["invitation"].endswith("Decision")
        ]
        if len(decision_dict) != 1:
            continue

        decision: str = decision_dict[0]["content"]["decision"]
        accepted: bool = "Accept" in decision
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
    """
    Collect reviews.

    :param weakness: Whether to include weakness or not
    """
    client = openreview.Client(baseurl="https://api.openreview.net")
    venues: List[str] = client.get_group(id="venues").members

    result = []
    for venue in tqdm(venues):
        subs_blind: List[openreview.Note] = client.get_all_notes(
            invitation=venue + "/-/Blind_Submission", details="directReplies"
        )
        subs_single: List[openreview.Note] = client.get_all_notes(
            invitation=venue + "/-/Submission", details="directReplies"
        )
        submissions: List[openreview.Note] = subs_blind + subs_single
        reviews: List[Dict[str, Union[str, bool]]] = get_reviews(
            venue, submissions, args.weaknesses
        )
        result.extend(reviews)

    filename = "reviews_with_weaknesses.pkl" if weakness else "reviews_without_weaknesses.pkl"

    pkl.dump(result, open(f"crawled/{filename}", "wb"))


def str_to_bool(value: str) -> bool:
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaknesses",
        "-w",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether crawl weaknesses or not",
    )
    args = parser.parse_args()
    main(args.weaknesses)
