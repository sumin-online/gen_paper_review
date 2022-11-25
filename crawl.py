import argparse
from glob import glob
from pathlib import Path
import pickle as pkl
import random
from time import sleep
from typing import Any, Dict

import requests
from tqdm import tqdm


def get_conference(venue: str, year: int) -> Dict[str, Any]:
    """
    Crawl list of paper submitted to venue-year.

    :param venue: conference name, str
    :param year: year, int
    :return res: metadata for all papers (accepted and rejected)

    Remark: whether a focal paper is accepted or not is manually added
    """
    print("\nGet conference metadata...", end=" ")
    url = f"https://api.openreview.net/notes?content.venue={venue}+{year}"
    url_rejected = url + "+Submitted"

    req_all = requests.get(url)
    res_all = req_all.json()
    for note in res_all["notes"]:
        note["accepted"] = True

    req_rejected = requests.get(url_rejected)
    res_rejected = req_rejected.json()
    for note in res_rejected["notes"]:
        note["accepted"] = False

    res = {
        "count": res_all["count"] + res_rejected["count"],
        "notes": res_all["notes"] + res_rejected["notes"],
    }
    print("done.")
    return res


def get_papers(venue: str, year: int, metadata: Dict[str, Any]) -> None:
    """
    Get paper details.

    :param venue: conference name, str
    :param year: year, int
    :param metadata: output of get_conference function, dict
    :return None

    Remark: There are some responses with 1 contents, indicating that there is no comments.
    """
    print("\nGet papers...")
    Path(f"{venue}_{year}/").mkdir(exist_ok=True)
    for note in tqdm(metadata["notes"]):
        url = f'https://api.openreview.net/notes?forum={note["id"]}'
        req = requests.get(url)
        res = req.json()
        note["comments"] = res
        pkl.dump(note, open(f'{venue}_{year}/{note["id"]}.pkl', "wb"))

        # prevent from banning
        sleep(random.uniform(1, 10))

    print("done.")


def aggregating_files(venue: str, year: int) -> None:
    """
    Aggregate paper review details.

    :param venue: conference name, str
    :param year: year, int
    :return None
    """
    Path("crawled").mkdir(exist_ok=True)
    res = []
    files = glob(f"{venue}_{year}/*.pkl")
    print("\nAggregating responses...", end=" ")
    for f in tqdm(files):
        temp = pkl.load(open(f, "rb"))
        res.append(temp)

    pkl.dump(res, open(f"crawled/{venue}_{year}.pkl", "wb"))
    print('done. \nResult file is automatically saved in "crawled" directory')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference", "-c", type=str, required=True, help="Crawling conference")
    parser.add_argument("--year", "-y", type=int, required=True, help="Conference year.")
    args = parser.parse_args()

    conference = get_conference(args.conference, args.year)
    get_papers(args.conference, args.year, conference)
    aggregating_files(args.conference, args.year)
