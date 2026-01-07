from __future__ import annotations

from typing import List, Optional, Sequence
from urllib.parse import quote_plus, urlparse

from lxml.html import fromstring
from playwright.async_api import async_playwright

# giữ lại googlesearch (nếu có thể dùng được) nhưng sẽ fallback
try:
    from googlesearch import search as google_search
except Exception:
    google_search = None


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _domain_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    d = _domain(url)
    allow = [a.lower() for a in allowed_domains]
    return any(d == a or d.endswith("." + a) for a in allow)


def _google_search_urls(query: str, limit: int) -> List[str]:
    if google_search is None:
        return []

    want = max(limit * 5, 10)
    try:
        it = google_search(query, num_results=want)
    except TypeError:
        it = google_search(query, num=want, stop=want)

    results: List[str] = []
    for u in it:
        if u not in results:
            results.append(u)
        if len(results) >= limit:
            break
    return results


async def _viblo_search_urls(query: str, limit: int, headless: bool = True) -> List[str]:
    """
    Search trực tiếp trên viblo.asia bằng Playwright (ổn định hơn Google).
    """
    q = quote_plus(query)
    search_url = f"https://viblo.asia/search?q={q}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(search_url, wait_until="domcontentloaded")
        html = await page.content()
        await browser.close()

    doc = fromstring(html)

    # Lấy các link bài viết dạng /p/...
    urls: List[str] = []
    for a in doc.xpath("//a[@href]"):
        href = a.get("href") or ""
        if href.startswith("/p/"):
            full = "https://viblo.asia" + href
        elif href.startswith("https://viblo.asia/p/"):
            full = href
        else:
            continue

        if full not in urls:
            urls.append(full)
        if len(urls) >= limit:
            break

    return urls


def search_urls(
    query: str,
    limit: int = 5,
    allowed_domains: Optional[Sequence[str]] = None,
) -> List[str]:
    allowed_domains = list(allowed_domains or [])

    # 1) Thử Google trước (nếu không bị chặn)
    urls = _google_search_urls(query, limit=limit)

    # lọc domain nếu cần
    if allowed_domains:
        urls = [u for u in urls if _domain_allowed(u, allowed_domains)]

    # 2) Fallback: nếu rỗng và đang muốn viblo.asia -> search thẳng Viblo bằng Playwright
    if not urls and (not allowed_domains or any(d.lower() == "viblo.asia" for d in allowed_domains)):
        import asyncio
        return asyncio.run(_viblo_search_urls(query, limit=limit, headless=True))

    return urls
