import os
import json
import logging
from typing import Annotated, List, Dict, Optional
from pydantic import Field
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

import telemetry

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"
METADATA_FILE = os.path.join(SCRAPE_DIR, "scraped_metadata.json")

# The pricing page for each provider we already know about. Without this the only
# way to reach a site is for the caller to paste its URL into the chat, so
# "go find the pricing for Groq" — a question with no URL in it — cannot be answered.
PROVIDERS_FILE = "providers.json"

# A pricing page older than this is worth scraping again before trusting it.
STALE_AFTER_DAYS = 1

# Firecrawl reads the page and returns these fields already typed, so the numbers
# never pass through anyone's prose on the way here.
PRICE_SCHEMA = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "unit": {"type": "string",
                             "enum": ["per_million_tokens", "per_hour", "other"]},
                    "input_price": {"type": ["number", "null"]},
                    "output_price": {"type": ["number", "null"]},
                },
                "required": ["model_name", "unit"],
            },
        }
    },
    "required": ["plans"],
}

PRICE_PROMPT = (
    "Extract the price list for RUNNING INFERENCE on this provider's models. "
    "Take rows only from a table that prices inference. Ignore any table under a "
    "heading about training, fine-tuning, deployment or GPU rental, even when its "
    "rows name models and quote per-million-token figures — those are a different "
    "product and cost several times what inference costs. If the page has no "
    "inference price table, return an empty list rather than the nearest table. "
    "For each row: model_name as written; unit set to per_million_tokens when the "
    "figure is a token price, per_hour for GPU or instance rental, otherwise other; "
    "input_price and output_price as the exact figures shown. If the columns are "
    "named something else (prefill and sample, prompt and completion), map the "
    "prompt-side column to input_price and the generation-side column to "
    "output_price. Only rows for a named model. Use null where a figure is absent."
)


def scrape_with_prices(app: FirecrawlApp, url: str, formats: List[str]) -> dict:
    """Fetch a page and its price table in one call, as a plain dict."""
    result = app.scrape(
        url,
        formats=list(formats) + [{'type': 'json', 'prompt': PRICE_PROMPT,
                                  'schema': PRICE_SCHEMA}],
    )
    # firecrawl-py v2 returns a pydantic Document; older versions a dict.
    return result.model_dump() if hasattr(result, "model_dump") else result


def save_content(provider_name: str, scrape_result: dict,
                 formats: List[str]) -> Dict[str, str]:
    """Write each requested format to disk, returning format -> filename."""
    content_files = {}
    for format_type in formats:
        content = scrape_result.get(format_type)
        if not content:
            logger.warning(f"No '{format_type}' content returned for {provider_name}")
            continue
        filename = f"{provider_name}_{format_type}.txt"
        with open(os.path.join(SCRAPE_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(content)
        content_files[format_type] = filename
    return content_files


def token_plans(scrape_result: dict) -> List[dict]:
    """The rows that price a named model both ways, per token.

    A GPU hourly rate is a different product, and a row with only half its prices
    is not something you can compare, so neither is worth passing on.
    """
    plans = (scrape_result.get('json') or {}).get('plans') or []
    return [p for p in plans
            if p.get('unit') == 'per_million_tokens'
            and p.get('input_price') is not None
            and p.get('output_price') is not None]


def provider_key(name: str) -> str:
    """Canonical spelling of a provider name, used as the storage key.

    Callers invent a new spelling every time — 'Fireworks', 'fireworks',
    'cloudrift ai', 'cloudrift-pricing-1' — and each one used to become its own
    entry, fragmenting the store and leaving extract_scraped_info to match
    whichever happened to come first.
    """
    return "_".join(str(name).strip().lower().split())


def _freshness(scraped_at: Optional[str]) -> Dict[str, object]:
    """How old a saved page is, so the caller does not have to do date maths.

    Models are unreliable at comparing timestamps, so the decision is made here
    and handed over as a plain boolean.
    """
    if not scraped_at:
        return {"age_days": None, "stale": True}

    age = datetime.now() - datetime.fromisoformat(scraped_at)
    age_days = round(age.total_seconds() / 86400, 2)
    return {"age_days": age_days, "stale": age_days >= STALE_AFTER_DAYS}


def _firecrawl(api_key: Optional[str] = None) -> FirecrawlApp:
    """Build a Firecrawl client, falling back to FIRECRAWL_API_KEY."""
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    return FirecrawlApp(api_key=api_key)


def _load_providers() -> Dict[str, str]:
    """The configured provider -> pricing-page URL map, keyed canonically.

    Returns an empty map when the file is missing or unreadable, so a caller that
    supplies its own URLs is never blocked by a broken config.
    """
    try:
        with open(PROVIDERS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {PROVIDERS_FILE}: {e}")
        return {}

    return {provider_key(name): entry["url"]
            for name, entry in (config.get("providers") or {}).items()
            if isinstance(entry, dict) and entry.get("url")}


def _load_display_names() -> Dict[str, str]:
    """Canonical key -> the spelling to show and store ('deepinfra' -> 'DeepInfra').

    Saved with the scrape so whatever writes to the database does not have to
    guess the casing and end up with one provider under two names.
    """
    try:
        with open(PROVIDERS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    return {provider_key(name): entry["display_name"]
            for name, entry in (config.get("providers") or {}).items()
            if isinstance(entry, dict) and entry.get("display_name")}


def _provider_aliases() -> Dict[str, str]:
    """Other spellings of a configured provider -> its canonical key.

    Questions name a company the way a person would — "CloudRift AI", "deep
    infra" — and that spelling has to reach the same stored entry as "cloudrift",
    or the same provider ends up saved twice under two keys.
    """
    try:
        with open(PROVIDERS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    aliases: Dict[str, str] = {}
    for name, entry in (config.get("providers") or {}).items():
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases") or []:
            aliases[provider_key(alias)] = provider_key(name)
    return aliases


def _match_provider(key: str, configured: Dict[str, str]) -> Optional[str]:
    """The configured provider a caller's spelling refers to, if exactly one does."""
    if key in configured:
        return key

    alias = _provider_aliases().get(key)
    if alias in configured:
        return alias

    # Last resort for a spelling nobody listed: "cloudrift-ai-pricing" still
    # points at cloudrift. Only when it points at one provider — an ambiguous
    # match is a wrong answer waiting to be stored.
    hits = [k for k in configured if k in key or key in k]
    return hits[0] if len(hits) == 1 else None


def _resolve_websites(websites: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Fill in the URLs the caller did not give.

    Nothing at all means every configured provider. A name mapped to an empty
    string means "the page we already know for this provider" — which is how a
    question that names a provider but no URL gets answered.
    """
    configured = _load_providers()

    if not websites:
        if not configured:
            raise ValueError(
                f"No websites given and no providers configured in {PROVIDERS_FILE}"
            )
        logger.info(f"No websites given; using the {len(configured)} configured provider(s)")
        return dict(configured)

    resolved: Dict[str, str] = {}
    for name, url in websites.items():
        key = provider_key(name)
        match = _match_provider(key, configured)

        # A configured provider keeps its canonical key even when the caller
        # supplies the URL, so "CloudRift AI" and "cloudrift" stay one entry.
        if match:
            key = match

        url = (url or "").strip() or configured.get(match or "", "")
        if not url:
            logger.error(f"No URL for '{name}' and none configured in {PROVIDERS_FILE}")
            continue
        resolved[key] = url

    if not resolved:
        raise ValueError(
            f"None of {list(websites)} could be resolved to a URL; "
            f"configured providers are {sorted(configured)}"
        )
    return resolved


def _load_metadata() -> Dict[str, dict]:
    """Load scraped_metadata.json, or an empty dict if it is missing or unreadable."""
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

mcp = FastMCP("llm_inference")
telemetry.setup("llm_inference")

# Named in the tool schema so the model knows which providers it can reach by name
# alone, instead of learning it from a failed call. Read once at startup; the
# server is spawned fresh per client run, so an edit to providers.json lands on
# the next run.
_CONFIGURED = sorted(_load_providers())
_WEBSITES_HELP = (
    "Provider name -> pricing page URL. The URL may be an empty string, which uses "
    "that provider's configured page; omit the argument entirely to scrape all of "
    "them. A URL given here always wins over the configured one. "
    + (f"Configured providers: {', '.join(_CONFIGURED)}."
       if _CONFIGURED else "No providers are configured, so URLs must be supplied.")
)

@mcp.tool()
@telemetry.traced("tool.scrape_websites", record=("websites", "formats"))
def scrape_websites(
    websites: Annotated[Optional[Dict[str, str]], Field(description=_WEBSITES_HELP)] = None,
    formats: List[str] = ['markdown'],
    refresh: bool = False,
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.

    Content is saved to disk and stays there, so a site only needs scraping once.
    Read an already-scraped site with extract_scraped_info instead of calling this
    again: each call spends Firecrawl credits and overwrites whatever was stored
    for that provider, which loses a good pricing page if the new URL is weaker.

    Args:
        websites: Dictionary of provider_name -> URL mappings. The URL is optional:
            pass an empty string to use the provider's configured pricing page
            (`{"groq": ""}`), and omit the argument entirely to scrape every
            provider in providers.json. Supplying a URL always overrides the
            configured one, so a caller can point at a page we do not know.
        formats: Formats to scrape and save (default: markdown only).
            Firecrawl charges the same one credit per page whichever formats are
            asked for, but html is roughly 85% of what a page weighs and every
            saved format is handed to the model by extract_scraped_info, so
            keeping html costs tokens on every turn and buys nothing that the
            markdown does not already say. Pass ['markdown', 'html'] if the
            raw html is genuinely needed.
        refresh: Scrape again even if the saved copy is still fresh (default False).
        api_key: Firecrawl API key (if None, expects environment variable)

    Returns:
        List of provider names whose content is available afterwards, whether it
        was fetched now or was already saved and still fresh.
    """

    websites = _resolve_websites(websites)

    app = _firecrawl(api_key)

    os.makedirs(SCRAPE_DIR, exist_ok=True)

    # Previous runs are kept, so re-scraping one provider does not wipe the others.
    scraped_metadata = _load_metadata()
    display_names = _load_display_names()
    successful_scrapes: List[str] = []

    for provider_name, url in websites.items():
        provider_name = provider_key(provider_name)

        # Already have this exact page and it has not gone stale: nothing to buy.
        saved = scraped_metadata.get(provider_name)
        if (not refresh and saved and saved.get('url') == url
                and saved.get('success') == 'true'
                and not _freshness(saved.get('scraped_at'))['stale']):
            logger.info(f"{provider_name}: saved copy is still fresh, not scraping again")
            successful_scrapes.append(provider_name)
            continue

        metadata = {
            "provider_name": provider_name,
            "display_name": display_names.get(provider_name, provider_name),
            "url": url,
            "domain": urlparse(url).netloc,
            "scraped_at": datetime.now().isoformat(),
            "formats": formats,
            "success": "false",
        }

        try:
            logger.info(f"Scraping {provider_name}: {url}")
            scrape_result = scrape_with_prices(app, url, formats)

            # v1 reports an explicit success flag; v2 only returns content on success.
            succeeded = scrape_result.get('success', any(scrape_result.get(f) for f in formats))

            if succeeded:
                content_files = save_content(provider_name, scrape_result, formats)

                # The caller can point a provider at a different URL than the one
                # already stored, and that URL can be the wrong page. Prices we
                # have are not worth trading for a page that carries none.
                plans = token_plans(scrape_result)
                if not plans and saved and saved.get('plan_count'):
                    logger.warning(
                        f"{provider_name}: {url} prices nothing, keeping the "
                        f"{saved['plan_count']} plan(s) already stored from "
                        f"{saved.get('url')}"
                    )
                    # `finally` writes `metadata` back, so restore through it
                    # rather than assigning here and having it overwritten.
                    metadata = saved
                    successful_scrapes.append(provider_name)
                    continue

                page_metadata = scrape_result.get('metadata') or {}
                metadata["success"] = "true"
                metadata["content_files"] = content_files
                if plans:
                    plans_file = f"{provider_name}_plans.json"
                    with open(os.path.join(SCRAPE_DIR, plans_file), 'w',
                              encoding='utf-8') as f:
                        json.dump(plans, f, indent=2)
                    metadata["plans_file"] = plans_file
                metadata["plan_count"] = len(plans)

                metadata["title"] = page_metadata.get("title", "")
                metadata["description"] = page_metadata.get("description", "")

                successful_scrapes.append(provider_name)
                logger.info(f"Saved {len(content_files)} file(s) for {provider_name}")
            else:
                error = scrape_result.get('error', 'unknown error')
                metadata["error"] = str(error)
                logger.error(f"Firecrawl failed for {provider_name}: {error}")

        except Exception as e:
            metadata["error"] = str(e)
            logger.error(f"Error scraping {provider_name} ({url}): {e}")

        finally:
            scraped_metadata[provider_name] = metadata

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(scraped_metadata, f, indent=4)

    logger.info(f"Scraped {len(successful_scrapes)}/{len(websites)} site(s): {successful_scrapes}")
    return successful_scrapes

@mcp.tool()
@telemetry.traced("tool.extract_scraped_info", record=("identifier",))
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.

    Args:
        identifier: The provider name, full URL, or domain to look for

    Returns:
        Formatted JSON string with the scraped information, including when the page
        was scraped (`scraped_at`), how old that is (`age_days`) and whether it is
        past its useful life (`stale`).
    """

    logger.info(f"Extracting information for identifier: {identifier}")
    if not os.path.isdir(SCRAPE_DIR):
        return f"There's no saved information related to identifier '{identifier}'."

    scraped_metadata = _load_metadata()
    needle = identifier.strip().lower()

    def candidates_for(metadata: dict, provider_name: str) -> List[str]:
        return [
            provider_name.lower(),
            str(metadata.get('url', '')).lower(),
            str(metadata.get('domain', '')).lower(),
        ]

    # Exact matches first, across every provider. Testing each provider in file
    # order and falling straight to the loose test hands 'groq_specdec' to 'groq',
    # because 'groq' is a substring of it and comes first — so the caller is given
    # a different page than the one it named.
    ordered = sorted(scraped_metadata.items(),
                     key=lambda kv: 0 if needle in candidates_for(kv[1], kv[0]) else 1)

    for provider_name, metadata in ordered:
        candidates = candidates_for(metadata, provider_name)
        # Matches the provider name, the full URL or the domain. Substring either
        # way round, so "cloudrift" still finds "cloudrift_ai".
        if any(c and (needle in c or c in needle) for c in candidates):
            result = metadata.copy()
            # Metadata saved before display_name existed still has to name the
            # provider the way the database spells it, so fill it in on read.
            result.setdefault(
                'display_name',
                _load_display_names().get(provider_name, provider_name)
            )
            result.update(_freshness(metadata.get('scraped_at')))

            # Typed price rows, when the scrape produced them, ride along with
            # the page text so the model can read figures without parsing tables.
            if metadata.get('plans_file'):
                plans_path = os.path.join(SCRAPE_DIR, metadata['plans_file'])
                if os.path.exists(plans_path):
                    with open(plans_path, 'r', encoding='utf-8') as f:
                        result['plans'] = json.load(f)

            if metadata.get('content_files'):
                result['content'] = {}
                try:
                    for format_type, filename in metadata['content_files'].items():
                        file_path = os.path.join(SCRAPE_DIR, filename)
                        with open(file_path, 'r', encoding='utf-8') as content_file:
                            result['content'][format_type] = content_file.read()
                except OSError as e:
                    logger.error(f"Could not read scraped content for '{identifier}': {e}")
                    return f"There's no saved information related to identifier '{identifier}'."

            logger.info(f"Matched identifier '{identifier}' to provider '{provider_name}'")
            return json.dumps(result, indent=2)

    logger.info(f"No match for identifier '{identifier}'")
    return f"There's no saved information related to identifier '{identifier}'."

if __name__ == "__main__":
    mcp.run(transport="stdio")
