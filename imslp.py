
import logging
import re
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from googlesearch import search


class IMSLP:
    IMSLP_BASE_URL = "https://imslp.org"

    logger: logging.Logger

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def find_imslp(self, query: str) -> Optional[str]:
        """Query Google for an IMSLP page.

        Args:
            query (str): The query to send Google.

        Returns:
            Optional[str]: The first imslp link in Google search results,
            or None if none was found.
        """
        try:
            self.logger.info(f"Googling '{query}' for imslp page.")
            for result in search(query):
                if str(result).startswith(self.IMSLP_BASE_URL):
                    return str(result)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.exception(f"Too many server requests: {e}")
                raise
        except Exception as e:
            logging.exception(f"IMSLP query {query} failed:\n\t{e}")
        return None

    COMPLETE_SCORE_RE = re.compile(r'^.*Complete Score.*$')

    def _extract_download_links(self, content: bytes) -> List[str]:
        links = list([])
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a"):
            if a.has_attr("rel") and "nofollow" in a["rel"]:
                span = a.find("span")
                if span and span["title"] == "Download this file":
                    if self.COMPLETE_SCORE_RE.match(span.text):
                        links.append(a["href"])
        return links

    # IMSLP_COOKIES = r'imslp_wikiLanguageSelectorLanguage=en; chatbase_anon_id=d8925c94-d976-492a-9649-e563f973d8a2; imslpdisclaimeraccepted=yes; __stripe_mid=5d13801d-837c-4919-8e35-88de460c440b313847; _gid=GA1.2.642930185.1737548859; __stripe_sid=d726e726-eeea-4292-b94d-715cac65d6979cf564; _ga_4QW4VCTZ4E=GS1.1.1737559129.13.1.1737560753.0.0.0; _ga=GA1.2.1606208118.1735899643; _ga_8370FT5CWW=GS1.2.1737559147.12.1.1737560755.0.0.0'
    IMSLP_COOKIES = {
        "imslp_wikiLanguageSelectorLanguage": "en",
        "chatbase_anon_id": "d8925c94-d976-492a-9649-e563f973d8a2",
        "imslpdisclaimeraccepted": "yes",
    }

    def find_pdf_links(self, imslp_page: str) -> List[str]:
        self.logger.info(f"Extracting download links from {imslp_page}")
        # Fetches the page and find the download link.
        response = requests.get(imslp_page)
        response.raise_for_status()

        return self._extract_download_links(response.content)

    def save_pdf(self, download_url: str, into: Path):
        self.logger.info(f"\tSaving {download_url} to {into}")
        try:
            response = requests.get(download_url)
            with open(into, "wb+") as fp:
                fp.write(response.content)
            return True
        except Exception as e:
            print(f"Failed to fetch url: {download_url}\n\t{e}")
            return False
