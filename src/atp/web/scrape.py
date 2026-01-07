from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from urllib.parse import urlparse

from lxml.html import fromstring
from playwright.async_api import async_playwright


@dataclass
class ScrapeResult:
    url: str
    html: str
    text: str


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _is_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    d = _domain(url)
    allow = [x.lower() for x in allowed_domains]
    return any(d == a or d.endswith("." + a) for a in allow)


def extract_text_from_html(html: str, content_selector: Optional[str] = None) -> str:
    doc = fromstring(html)

    # bỏ các phần không cần thiết
    for bad in doc.xpath("//script|//style|//noscript"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)

    if content_selector:
        # Lấy đúng vùng nội dung theo CSS selector (ví dụ: "article")
        nodes = doc.cssselect(content_selector)
        if nodes:
            text = "\n".join(n.text_content() for n in nodes)
        else:
            # fallback: lấy toàn trang nếu selector không match
            text = doc.text_content()
    else:
        text = doc.text_content()

    # normalize whitespace
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


async def scrape_url(
    url: str,
    allowed_domains: Optional[Sequence[str]] = None,
    headless: bool = True,
    timeout_ms: int = 30000,
    content_selector: Optional[str] = None,
) -> ScrapeResult:
    if allowed_domains and not _is_allowed(url, allowed_domains):
        raise ValueError(f"Domain not allowed: {_domain(url)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        page.set_default_timeout(timeout_ms)

        await page.goto(url, wait_until="domcontentloaded")
        html = await page.content()

        await browser.close()

    text = extract_text_from_html(html, content_selector=content_selector)
    return ScrapeResult(url=url, html=html, text=text)
