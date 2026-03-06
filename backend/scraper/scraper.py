"""Scraper for Hugging Face trending papers."""

from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from lxml import etree


class Paper(TypedDict):
    """Type definition for paper information."""

    title: str
    url: str
    arxiv_url: str | None
    github_url: str | None
    github_stars: str | None
    author: str | None
    published: str | None
    abstract: str


def fetchTrendingPapers() -> list[Paper]:
    """
    Retrieve paper information from the Hugging Face trending page.

    Returns:
        A list of paper information. Each element is a dictionary with the following keys:
        - title: Paper title
        - url: URL of the paper page on Hugging Face
        - arxiv_url: arXiv URL (None if not available)
        - github_url: GitHub URL (None if not available)
        - github_stars: Number of GitHub stars (None if not available)
        - author: Author name (None if not available)
        - published: Publication date (None if not available)
        - abstract: Abstract of the paper

    Raises:
        requests.RequestException: HTTP communication error
        Exception: HTML parsing error
    """
    url = "https://huggingface.co/papers/trending"

    # Fetch the page from Hugging Face
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Parse using BeautifulSoup and lxml
    soup = BeautifulSoup(response.content, "lxml")
    dom = etree.HTML(str(soup))

    papers: list[Paper] = []

    # Retrieve articles sequentially starting from index 1 (assume up to ~50 items)
    for article_num in range(1, 51):
        try:
            # Title and URL
            title_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[2]/h3/a"
            title_elements = dom.xpath(title_xpath)
            if not title_elements:
                # Stop if there are no more articles
                break

            title_element = title_elements[0]
            title = title_element.text.strip() if title_element.text else ""
            href = title_element.get("href", "")
            paper_url = f"https://huggingface.co{href}" if href.startswith("/") else href

            # Abstract
            abstract_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[2]/div[1]/p"
            abstract_elements = dom.xpath(abstract_xpath)
            abstract = (
                abstract_elements[0].text.strip()
                if abstract_elements and abstract_elements[0].text
                else ""
            )

            # arXiv URL
            arxiv_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[3]/a[2]"
            arxiv_elements = dom.xpath(arxiv_xpath)
            arxiv_url = None
            if arxiv_elements:
                arxiv_href = arxiv_elements[0].get("href", "")
                if "arxiv.org" in arxiv_href:
                    arxiv_url = arxiv_href

            # GitHub URL
            github_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[3]/a[1]"
            github_elements = dom.xpath(github_xpath)
            github_url = None
            if github_elements:
                github_href = github_elements[0].get("href", "")
                if "github.com" in github_href:
                    github_url = github_href

            # GitHub Stars
            github_stars = None
            if github_url:
                stars_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[3]/a[1]/span[2]/span"
                stars_elements = dom.xpath(stars_xpath)
                if stars_elements and stars_elements[0].text:
                    github_stars = stars_elements[0].text.strip()

            # Author
            author_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[2]/div[2]/a/span"
            author_elements = dom.xpath(author_xpath)
            author = None
            if author_elements and author_elements[0].text:
                author = author_elements[0].text.strip()

            # Published date
            published_xpath = f"/html/body/div[1]/main/div[2]/section/div[2]/article[{article_num}]/div[1]/div[2]/div[2]/span[2]"
            published_elements = dom.xpath(published_xpath)
            published = None
            if published_elements and published_elements[0].text:
                published_text = published_elements[0].text.strip()
                # "Published on Oct 1, 2025" -> "Oct 1, 2025"
                if "Published on " in published_text:
                    published = published_text.replace("Published on ", "")
                else:
                    published = published_text

            paper: Paper = {
                "title": title,
                "url": paper_url,
                "arxiv_url": arxiv_url,
                "github_url": github_url,
                "github_stars": github_stars,
                "author": author,
                "published": published,
                "abstract": abstract,
            }
            papers.append(paper)

        except Exception as e:
            # Ignore errors when retrieving individual articles and continue
            print(f"Warning: Failed to parse article {article_num}: {e}")
            continue

    return papers