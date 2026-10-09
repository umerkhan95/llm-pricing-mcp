import ast
import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import timedelta
from pathlib import Path
import re

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Keep SDK transport details out of the interactive chatbot output. API errors
# still propagate through the client and are logged by the application.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

telemetry.setup("client")

MODEL_NAME = "claude-sonnet-4-5-20250929"

# Most questions are answered in two or three tool calls. Hunting a price that a
# provider never published can otherwise run for dozens of rounds, each one a
# scrape and an LLM turn, and still end up where it started.
MAX_TOOL_CALLS_PER_QUERY = 8

SYSTEM_PROMPT = (
    "You answer LLM inference pricing questions from scraped pages and a SQLite table.\n"
    "\n"
    "DATA\n"
    "- Table pricing_plans: company_name, plan_name, input_tokens, output_tokens, "
    "currency, billing_period, scraped_at, source_url. input_tokens/output_tokens hold "
    "PRICES in USD per million tokens - say 'per million tokens', never 'per token'.\n"
    "- Answer from read_query. If the table has nothing, read the saved page with "
    "extract_scraped_info before concluding anything is missing. Do not ask permission.\n"
    "- Scrape only when a page was never scraped, came back stale, or the user asks. "
    "Reuse the URL already stored for that provider.\n"
    "- When the user only asks to scrape, call scrape_websites once, then reply in one "
    "line with how many of the requested sites were scraped and which; look up nothing else.\n"
    "- Match loosely: LOWER(company_name) LIKE '%cloudrift%', and "
    "LOWER(REPLACE(plan_name,'-',' ')) LIKE '%deepseek v3%' for models.\n"
    "- Before judging whether a model matches, rewrite the requested name and each "
    "returned plan_name in lower case with hyphens replaced by spaces, then compare "
    "those. Equal means it IS the requested model: answer with it directly and do not "
    "use the missing-model wording. 'deepseek-v3' and 'DeepSeek V3' rewrite to the same "
    "string and are one model; 'deepseek v3.1' rewrites differently and is another.\n"
    "\n"
    "CITATIONS\n"
    "- Always SELECT plan_name, source_url and scraped_at alongside the prices, and end "
    "with 'Source:' and "
    "'Last updated:' lines copied character for character from the returned rows - keep "
    "the full timestamp, never shorten it to a date or reformat it. One pair per "
    "provider in a comparison. Never invent either; write 'Unavailable' only when a "
    "returned row holds NULL.\n"
    "\n"
    "MISSING MODEL\n"
    "- Call a model missing only after reading plan_name values back; rows without "
    "plan_name prove nothing. Every model you name must appear in the returned rows - "
    "never invent one to fill a section.\n"
    "- Never print a provider as a block of Unavailable or N/A fields. Write one line - "
    "'<provider> does not list <model> in the saved data.' - then a section headed "
    "'Closest available from <provider>' giving that provider's nearest stored model "
    "under its real plan name with its prices, source and last-updated, and a short line "
    "on why it is nearest and that a different model makes this indicative only. That "
    "section is required unless the provider has no stored models at all.\n"
    "- Never call one model cheaper or more expensive than another unless both prices "
    "belong to the same model.\n"
    "\n"
    "Use one name per provider for the whole conversation; reuse the stored name and URL."
)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


# Bump when normalize_key changes — existing rows are then re-keyed on startup.
NORMALIZATION_VERSION = "3"

# Filler the model sprinkles into plan names inconsistently. The same GPU shows up
# as "H100 80GB", "H100 80GB On-Demand", "On-Demand H100" and "Rental H100 80GB"
# across runs; dropping these words keeps all four on one row. Words that
# distinguish real products (serverless, dedicated, cached, model numbers) stay.
_KEY_NOISE = {
    "gpu", "gpus", "instance", "instances", "plan", "plans", "tier", "tiers",
    "model", "models", "pricing", "price", "prices", "cost", "costs",
    "on", "demand", "ondemand", "rental", "rent", "managed", "deployment",
    "deployments", "param", "params", "parameters", "up", "to", "over", "from",
    "per", "hour", "hourly", "the", "a", "an", "and",
}


# What is left of a plan name once normalize_key has stripped its filler. A name
# made only of these describes how a provider bills - "Pay-per-token pricing
# model", "Unknown Plan" - rather than naming something you can buy.
_GENERIC_PLAN_TOKENS = {
    "pay", "usage", "token", "based", "structure", "detail", "information",
    "standard", "general", "default", "unknown", "none", "various", "multiple",
}


def normalize_key(value: Any) -> str:
    """Canonical form of a company or plan name, used only for de-duplication.

    Lowercases, drops punctuation and filler words, then sorts the remaining
    tokens so word order does not matter: 'DeepSeek-V3.1', 'DeepSeek V3.1' and
    'deepseek v3.1' collapse to one key, as do 'H100 80GB On-Demand' and
    'Rental H100 80GB'. 'DeepSeek-V3' and 'DeepSeek-V3-0324' stay distinct.
    The original text is still stored for display.
    """
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    # Keep digits and dots so version numbers such as v3.1 survive intact.
    text = re.sub(r"[^a-z0-9.]+", " ", text)

    tokens = [t.strip(".") for t in text.split()]
    tokens = [t for t in tokens if t and t not in _KEY_NOISE]
    # Singular/plural drift: "Custom LLMs - H100" vs "Custom LLM - H100".
    # Only on alphabetic tokens, so "80gb" and version numbers are untouched.
    tokens = [
        t[:-1] if len(t) > 3 and t.isalpha() and t.endswith("s") and not t.endswith("ss") else t
        for t in tokens
    ]

    # Sorting makes "qwen3 8b embeddings" and "embeddings qwen3 8b" the same key.
    return " ".join(sorted(tokens))


def result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into plain text."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and "result" in structured:
        value = structured["result"]
        return value if isinstance(value, str) else json.dumps(value)

    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            parts.append(text if text is not None else str(block))
        return "\n".join(parts)
    return str(content)


def parse_rows(text: str) -> list[dict]:
    """Parse rows from mcp-server-sqlite, which returns a Python literal, not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def sql_quote(value: Any) -> str:
    """Escape a value for inlining into a SQL string literal.

    mcp-server-sqlite takes a raw query string with no parameter binding, so
    every inlined value has to be escaped here — a plan name like
    "DeepSeek's V3" would otherwise terminate the literal early.
    """
    return str(value).replace("'", "''")


# Per-million-token prices sit well under this. A larger number is a misread,
# not a price: "PRICE PER 1M TOKENS" has been stored as 1000000 before now.
MAX_PLAUSIBLE_PRICE = 10_000


def sql_number(value: Any) -> str:
    """Render a price for inlining into SQL: a bare number, or NULL if missing.

    The model returns null for plans that have no per-token price at all —
    hourly GPU rentals and training tiers. Quoting those stored the text
    'None' in a REAL column, which then printed as "$None". Values outside a
    plausible price range are dropped the same way, so a misread column header
    does not end up in the table as a real figure.
    """
    if isinstance(value, bool):
        return "NULL"
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value or "").strip().lstrip("$").replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", text):
            return "NULL"
        number = float(text)

    if not 0 <= number <= MAX_PLAUSIBLE_PRICE:
        logging.warning(f"Ignoring implausible price: {value!r}")
        return "NULL"
    # Always emit the parsed number, never the original text: "$0.25" and
    # "1,200" are valid input but not valid SQL.
    return str(int(number)) if number.is_integer() else repr(number)


def sql_text(value: Any) -> str:
    """Render a text value for inlining into SQL, or NULL when there isn't one."""
    if value is None or str(value).strip() == "":
        return "NULL"
    return "'" + sql_quote(value) + "'"


def price_in_source(value: Any, source_text: str) -> bool:
    """Is this figure actually written in the pages we scraped?

    A small model will invent a plausible price and then invent a quote to
    support it — $0.79 for a model whose page says "Contact Sales". Nothing in
    the prompt reliably stops that, but the number either appears in the source
    or it does not, and that is cheap to check.

    A missing price passes: there is nothing to disagree with. A price with no
    evidence behind it does not, because the alternative is to believe whatever
    came back when every lookup failed.
    """
    if value is None:
        return True
    if not source_text:
        return False

    number = sql_number(value)
    if number == "NULL":
        return True

    as_float = float(number)
    # 0.9 may be written as "0.9" or "0.90"; 12 as "12" or "12.00".
    variants = {number, f"{as_float:g}", f"{as_float:.2f}", f"{as_float:.3f}".rstrip("0")}
    # Bounded, or "0.9" would match inside "$0.99" and wave the invention through.
    return any(re.search(r"(?<![\d.])" + re.escape(v) + r"(?!\d)", source_text)
               for v in variants)


def format_price(value: Any) -> str:
    """Format a stored price for display, or 'n/a' when there isn't one."""
    if value is None or str(value).strip() in ("", "None"):
        return "n/a"
    return f"${value}"


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file {file_path}: {e}")
            raise

        if "mcpServers" not in config:
            raise ValueError(f"Configuration file {file_path} is missing the 'mcpServers' key")

        return config

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")

        tools_response = await self.session.list_tools()
        tools: List[ToolDefinition] = []

        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            tools.append(tool_def)

        return tools

    @telemetry.traced("tool.execute", record=("tool_name", "arguments"))
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=60),
                )
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool {tool_name}: {e}. Attempt {attempt} of {retries}."
                )
                if attempt >= retries:
                    logging.error(f"Max retries reached for {tool_name}. Failing.")
                    raise
                logging.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client

    async def _write(self, query: str) -> Any:
        """Run a non-SELECT statement against the SQLite MCP server."""
        return await self.sqlite_server.execute_tool("write_query", {"query": query})

    async def _read(self, query: str) -> list[dict]:
        """Run a SELECT and return its rows."""
        result = await self.sqlite_server.execute_tool("read_query", {"query": query})
        return parse_rows(result_to_text(result))

    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:

            await self._write("""
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Rows written before sql_number existed hold the text 'None'.
            for column in ("input_tokens", "output_tokens"):
                await self._write(
                    f"UPDATE pricing_plans SET {column} = NULL "
                    f"WHERE {column} IN ('None', '')"
                )

            await self._migrate_to_normalized_keys()

            logging.info("✓ Data extraction tables initialized")

        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _migrate_to_normalized_keys(self) -> None:
        """Bring the table up to date with the current normalize_key rules.

        De-duplication runs on normalized keys rather than the raw names, because
        the model spells the same plan differently between runs.
        """
        # Ask first. Adding a column that exists is only reported as an error, but
        # the sqlite server prints that error on every start after the first.
        # mcp-server-sqlite only permits SELECT statements through read_query;
        # the table-valued pragma exposes the same schema information safely.
        existing = {row["name"] for row in
                    await self._read(
                        "SELECT name FROM pragma_table_info('pricing_plans')"
                    )}
        for column in ("company_key", "plan_key", "scraped_at", "source_url"):
            if column not in existing:
                await self._write(f"ALTER TABLE pricing_plans ADD COLUMN {column} TEXT")

        # Indexes come down before re-keying, or the UPDATEs below collide.
        # idx_pricing_plans_company_plan is an older index keyed on raw names.
        await self._write("DROP INDEX IF EXISTS idx_pricing_plans_company_plan")
        await self._write("DROP INDEX IF EXISTS idx_pricing_plans_keys")

        await self._reset_keys_if_normalization_changed()
        await self._backfill_keys()

        # Collapse duplicates left by earlier runs, keeping the newest row for
        # each plan, so the unique index can be created.
        await self._write("""
            DELETE FROM pricing_plans
            WHERE id NOT IN (
                SELECT MAX(id) FROM pricing_plans GROUP BY company_key, plan_key
            )
        """)

        # One row per (company, plan) — re-asking the same question now refreshes
        # the prices instead of appending another copy.
        await self._write("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pricing_plans_keys
            ON pricing_plans (company_key, plan_key)
        """)

    async def _reset_keys_if_normalization_changed(self) -> None:
        """Clear stored keys when normalize_key changed, so they get recomputed."""
        await self._write(
            "CREATE TABLE IF NOT EXISTS extraction_meta (key TEXT PRIMARY KEY, value TEXT)"
        )

        rows = await self._read(
            "SELECT value FROM extraction_meta WHERE key = 'normalization_version'"
        )
        stored_version = rows[0]["value"] if rows else None

        if stored_version == NORMALIZATION_VERSION:
            return

        await self._write("UPDATE pricing_plans SET company_key = NULL, plan_key = NULL")
        await self._write(
            "INSERT INTO extraction_meta (key, value) "
            f"VALUES ('normalization_version', '{NORMALIZATION_VERSION}') "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
        )
        if stored_version is not None:
            logging.info(
                f"Normalization changed ({stored_version} -> {NORMALIZATION_VERSION}); re-keying rows"
            )

    async def _backfill_keys(self) -> None:
        """Fill company_key/plan_key for rows that do not have them yet."""
        rows = await self._read(
            "SELECT id, company_name, plan_name FROM pricing_plans WHERE plan_key IS NULL"
        )

        for row in rows:
            await self._write(
                "UPDATE pricing_plans SET "
                f"company_key = '{sql_quote(normalize_key(row['company_name']))}', "
                f"plan_key = '{sql_quote(normalize_key(row['plan_name']))}' "
                f"WHERE id = {int(row['id'])}"
            )

        if rows:
            logging.info(f"Backfilled normalized keys for {len(rows)} existing row(s)")

    @telemetry.traced("llm.extraction", record=("prompt",))
    async def store_scraped_plans(self, providers: list[str], scraper_server: Server,
                                  user_query: str) -> int:
        """Write the scraper's typed price rows to the database, with no model involved.

        Firecrawl already returns model_name and the two prices as typed fields.
        Putting an extraction pass in front of them only adds somewhere for the
        figures to drift, and they did: 'Kimi K3 US' reached the table as
        3.3/16.5 while the scrape that produced it says 4.5/22.5.

        Returns the number of rows written.
        """
        stored = 0
        for provider in providers:
            try:
                result = await scraper_server.execute_tool(
                    "extract_scraped_info", {"identifier": provider}
                )
            except Exception as e:
                logging.error(f"Could not read back '{provider}' after scraping: {e}")
                continue

            try:
                record = json.loads(result_to_text(result))
            except json.JSONDecodeError:
                # extract_scraped_info answers in plain text when it has nothing.
                logging.warning(f"No saved information to store for '{provider}'")
                continue

            # display_name is written by the scraper from providers.json; the
            # fallbacks cover metadata saved before that field existed.
            company = (record.get("display_name")
                       or record.get("provider_name")
                       or provider)
            written = await self._store_plan_rows(company, record, user_query)
            logging.info(f"{company}: stored {written} priced plan(s) from the scrape")
            stored += written

        return stored

    async def _store_plan_rows(self, company: str, record: dict, user_query: str) -> int:
        """Upsert one provider's price rows. Every value here came off the page."""
        source_url = record.get("url")
        scraped_at = record.get("scraped_at")
        rows = []

        for plan in record.get("plans") or []:
            # Only per-token rows compare against each other. A GPU hour or a
            # price per million characters is a different product on the same page.
            if plan.get("unit") != "per_million_tokens":
                continue

            plan_name = str(plan.get("model_name") or "").strip()
            if not plan_name:
                continue

            input_price = plan.get("input_price")
            output_price = plan.get("output_price")
            if input_price is None and output_price is None:
                continue

            rows.append(f"""(
                    '{sql_quote(company)}',
                    '{sql_quote(plan_name)}',
                    '{sql_quote(normalize_key(company))}',
                    '{sql_quote(normalize_key(plan_name))}',
                    {sql_number(input_price)},
                    {sql_number(output_price)},
                    'USD',
                    'per million tokens',
                    '[]',
                    '',
                    '{sql_quote(user_query)}',
                    {sql_text(scraped_at)},
                    {sql_text(source_url)}
                )""")

        if not rows:
            return 0

        # One statement per provider rather than one per plan: a pricing page
        # carries dozens of models, and each write is a round trip to the server.
        await self._write(f"""
            INSERT INTO pricing_plans (
                company_name, plan_name, company_key, plan_key,
                input_tokens, output_tokens, currency,
                billing_period, features, limitations, source_query,
                scraped_at, source_url
            )
            VALUES {", ".join(rows)}
            ON CONFLICT (company_key, plan_key) DO UPDATE SET
                -- The scrape is the better source, so unlike the extraction
                -- path this overwrites rather than coalescing: a price that
                -- has come off the page is not worth keeping once the page
                -- has been read again.
                company_name   = excluded.company_name,
                plan_name      = excluded.plan_name,
                input_tokens   = excluded.input_tokens,
                output_tokens  = excluded.output_tokens,
                currency       = excluded.currency,
                billing_period = excluded.billing_period,
                source_query   = excluded.source_query,
                scraped_at     = excluded.scraped_at,
                source_url     = excluded.source_url,
                created_at     = CURRENT_TIMESTAMP
        """)
        return len(rows)

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                # A multi-provider comparison overflows 1024 tokens and the JSON
                # comes back truncated, so give the extraction room.
                max_tokens=8192,
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )

            text_content = ""
            for content in response.content:
                if content.type == 'text':
                    text_content += content.text

            return text_content.strip()

        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    @telemetry.traced("store.pricing_plans", record=("user_query", "source_url", "scraped_at"))
    async def extract_and_store_data(self, user_query: str, llm_response: str,
                                   source_url: str = None,
                                   scraped_at: str = None,
                                   source_text: str = "") -> None:
        """Extract structured data from LLM response and store it."""
        try:
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:

            Text: {llm_response}

            The text may cover several providers. Return one entry per provider
            with this structure:
            {{
                "companies": [
                    {{
                        "company_name": "company name",
                        "plans": [
                            {{
                                "plan_name": "plan name",
                                "input_tokens": number or null,
                                "output_tokens": number or null,
                                "currency": "USD",
                                "billing_period": "per-token, per-hour or one-time",
                                "features": ["feature1", "feature2"],
                                "limitations": "any limitations mentioned",
                                "query": "the user's query"
                            }}
                        ]
                    }}
                ]
            }}

            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")

            try:
                pricing_data = json.loads(extraction_response)
            except json.JSONDecodeError as e:
                logging.error(
                    f"Extraction returned invalid JSON ({e}). "
                    f"Response ended with: ...{extraction_response[-120:]!r}"
                )
                return

            # Accept the single-company shape too, in case the model falls back to it.
            companies = pricing_data.get("companies")
            if companies is None:
                companies = [pricing_data] if pricing_data.get("plans") else []

            stored = 0
            for company in companies:
                company_name = company.get("company_name", "Unknown")

                for plan in company.get("plans", []):
                    plan_name = plan.get("plan_name", "Unknown Plan")

                    # A plan named after its own company means the extractor found
                    # a price but no model to attach it to. Storing it invents a
                    # product that the provider does not sell.
                    if normalize_key(plan_name) == normalize_key(company_name):
                        logging.warning(
                            f"Skipping plan named after its company: {company_name!r}"
                        )
                        continue

                    # Same idea, different shape: a name that says only how the
                    # billing works names no product either.
                    tokens = normalize_key(plan_name).split()
                    if not tokens or set(tokens) <= _GENERIC_PLAN_TOKENS:
                        logging.warning(f"Skipping generic plan name: {plan_name!r}")
                        continue

                    # Drop any figure that is not in the scraped text. The row is
                    # still worth keeping — the plan exists — but an invented
                    # price is worse than no price.
                    input_price = plan.get("input_tokens")
                    output_price = plan.get("output_tokens")
                    for label, price in (("input", input_price), ("output", output_price)):
                        if not price_in_source(price, source_text):
                            logging.warning(
                                f"{company_name} / {plan_name}: {label} price {price!r} "
                                f"is not in the scraped text, storing NULL"
                            )
                            if label == "input":
                                input_price = None
                            else:
                                output_price = None

                    # Page titles and marketing phrases — "CloudRift AI Inference",
                    # "Turbo Tier" — arrive as plans with no figures attached.
                    # Whatever they are, they are not rows of a price table.
                    if input_price is None and output_price is None:
                        logging.warning(
                            f"Skipping {company_name} / {plan_name}: no price"
                        )
                        continue

                    await self._write(f"""
                        INSERT INTO pricing_plans (
                            company_name, plan_name, company_key, plan_key,
                            input_tokens, output_tokens, currency,
                            billing_period, features, limitations, source_query,
                            scraped_at, source_url
                        )
                        VALUES (
                            '{sql_quote(company_name)}',
                            '{sql_quote(plan_name)}',
                            '{sql_quote(normalize_key(company_name))}',
                            '{sql_quote(normalize_key(plan_name))}',
                            {sql_number(input_price)},
                            {sql_number(output_price)},
                            '{sql_quote(plan.get("currency", "USD"))}',
                            '{sql_quote(plan.get("billing_period", "unknown"))}',
                            '{sql_quote(json.dumps(plan.get("features", [])))}',
                            '{sql_quote(plan.get("limitations", ""))}',
                            '{sql_quote(user_query)}',
                            {sql_text(scraped_at)},
                            {sql_text(source_url)}
                        )
                        ON CONFLICT (company_key, plan_key) DO UPDATE SET
                            -- Keep a price we already have when the new row has
                            -- none: the plan gets mentioned again in a later answer
                            -- whose sources do not carry the figure, and that
                            -- mention must not erase what was grounded earlier.
                            input_tokens   = COALESCE(excluded.input_tokens, input_tokens),
                            output_tokens  = COALESCE(excluded.output_tokens, output_tokens),
                            currency       = excluded.currency,
                            billing_period = excluded.billing_period,
                            features       = excluded.features,
                            limitations    = excluded.limitations,
                            source_query   = excluded.source_query,
                            scraped_at     = COALESCE(excluded.scraped_at, scraped_at),
                            source_url     = COALESCE(excluded.source_url, source_url),
                            created_at     = CURRENT_TIMESTAMP
                    """)
                    stored += 1

            names = ", ".join(c.get("company_name", "Unknown") for c in companies)
            logger.info(f"Stored {stored} pricing plans across {len(companies)} provider(s): {names}")

        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, Server] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def _ask(self, messages: list) -> Any:
        """One round trip to the model, recorded in the event log."""
        started = time.perf_counter()
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model=MODEL_NAME,
            system=SYSTEM_PROMPT,
            tools=self.available_tools,
            # One tool per turn: the loop below pairs each tool_use with a single
            # tool_result, so a turn carrying two of them would leave one unanswered.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        )
        telemetry.log_llm_call(MODEL_NAME, messages, self.available_tools,
                               SYSTEM_PROMPT, response,
                               (time.perf_counter() - started) * 1000)
        return response

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        telemetry.set_query(query)
        messages = [{'role': 'user', 'content': query}]
        response = self._ask(messages)

        full_response = ""
        # Every page this answer drew on. Collected rather than latched: the first
        # URL seen is not the source of every row, and stamping it on all of them
        # cited cloudrift.ai as the source of DeepInfra's DeepSeek-V3 price.
        source_urls: list[str] = []
        scraped_ats: list[str] = []
        source_text = []   # what the tools actually returned, to check prices against
        stored_rows = []   # read_query results: prices already in the table
        scraped = False

        tool_calls = 0

        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    full_response += content.text + "\n"
                    assistant_content.append(content)

                    # A text-only response means the model is done using tools.
                    if len(response.content) == 1:
                        process_query = False

                elif content.type == 'tool_use':
                    tool_calls += 1
                    if tool_calls > MAX_TOOL_CALLS_PER_QUERY:
                        logging.warning(
                            f"Stopping after {MAX_TOOL_CALLS_PER_QUERY} tool calls; "
                            f"answering with what has been found so far"
                        )
                        process_query = False
                        break

                    assistant_content.append(content)
                    messages.append({'role': 'assistant', 'content': assistant_content})

                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    if tool_name == "scrape_websites" and re.search(
                        r"\b(scrape|rescrape|re-scrape|refresh|fresh)\b",
                        query,
                        re.IGNORECASE,
                    ):
                        tool_args = {**tool_args, "refresh": True}
                    print(f"\n[Calling tool {tool_name} with args {tool_args}]\n")

                    # Route the call to whichever server exposes this tool.
                    server = self.tool_to_server.get(tool_name)

                    if server is None:
                        result_text = f"Error: no server found for tool '{tool_name}'"
                        logging.error(result_text)
                    else:
                        try:
                            result = await server.execute_tool(tool_name, tool_args)
                            result_text = result_to_text(result)
                        except Exception as e:
                            result_text = f"Error executing tool {tool_name}: {e}"
                            logging.error(result_text)
                        else:
                            # A finished scrape goes into the database here, off
                            # the typed rows, rather than waiting for the answer
                            # to be written and read back out again. Kept out of
                            # the try above so a database problem is not reported
                            # to the model as a failed scrape.
                            if tool_name == "scrape_websites":
                                await self._store_scrape(result_text, server, query, tool_args)

                    url = self._extract_url_from_result(result_text)
                    if url:
                        source_urls.append(url)
                    scraped_ats.extend(self._extract_scraped_at_from_result(result_text))
                    # Pages are evidence for a new figure. Stored rows are kept
                    # apart: they may vouch only for a price the table already
                    # holds, which re-writing leaves unchanged.
                    if tool_name in ("extract_scraped_info", "scrape_websites"):
                        source_text.append(result_text)
                    elif tool_name == "read_query":
                        stored_rows.append(result_text)
                    if tool_name == "scrape_websites":
                        scraped = True

                    messages.append({
                        'role': 'user',
                        'content': [{
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': result_text
                        }]
                    })

                    response = self._ask(messages)

                    # If the follow-up is plain text, the model has finished.
                    if len(response.content) == 1 and response.content[0].type == 'text':
                        print(response.content[0].text)
                        full_response += response.content[0].text + "\n"
                        process_query = False

        # Answers based only on read_query already use rows persisted in SQLite,
        # and a scrape has just written its rows deterministically; neither needs
        # the prose extractor. It runs only when the answer drew on a saved page.
        if self.data_extractor and full_response.strip() and source_text and not scraped:
            # Cite a page only when exactly one was read. With several there is no
            # way to tell which row came from which, and a confident wrong citation
            # is worse than an absent one; the deterministic path has already
            # written the right URL per provider.
            def only(seen: list[str]) -> str | None:
                return seen[0] if len(set(seen)) == 1 else None

            await self.data_extractor.extract_and_store_data(
                query, full_response.strip(), only(source_urls),
                only(scraped_ats), "\n".join(source_text + stored_rows)
            )

    async def _store_scrape(self, result_text: str, scraper_server: Server,
                            query: str, tool_args: dict) -> None:
        """Push a finished scrape into the database, deterministically.

        scrape_websites returns the provider names it has content for; those are
        read back as typed rows and written straight to SQLite. Failures are
        logged and swallowed — the answer to the user does not depend on this.
        """
        if not self.data_extractor:
            return

        try:
            providers = json.loads(result_text)
        except json.JSONDecodeError:
            logging.warning("Scrape result was not JSON; nothing stored")
            return

        # FastMCP returns a bare list, but wraps it as {"result": [...]} when
        # structured output is on.
        if isinstance(providers, dict):
            providers = providers.get("result")
        if not isinstance(providers, list):
            return

        requested = tool_args.get("websites") or {}
        if requested:
            print(f"\nSuccessfully scraped {len(providers)} out of {len(requested)} websites")
        if not providers:
            return

        try:
            stored = await self.data_extractor.store_scraped_plans(
                [str(p) for p in providers], scraper_server, query
            )
            print(f"\n[Stored {stored} priced plan(s) from the scrape]\n")
        except Exception as e:
            logging.error(f"Could not store scraped plans: {e}")

    def _extract_scraped_at_from_result(self, result_text: str) -> list[str]:
        """Pull every scrape timestamp out of a tool result."""
        # Pages arrive as JSON; read_query rows as a Python literal with single
        # quotes. Both count: a stamp missed here gets another provider's stamp.
        stamps = re.findall(r'''["']scraped_at["']\s*:\s*["']([^"']+)["']''', result_text)
        return stamps

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data()
                    continue

                await self.process_query(query)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return

        try:
            pricing = await self.sqlite_server.execute_tool("read_query", {
                "query": (
                    "SELECT company_name, plan_name, input_tokens, output_tokens, "
                    "currency, scraped_at, source_url "
                    # id breaks ties — created_at only has second precision
                    "FROM pricing_plans ORDER BY created_at DESC, id DESC LIMIT 5"
                )
            })

            print("\nRecently Stored Data:")
            print("=" * 50)
            print("\nPricing Plans:")

            rows = parse_rows(result_to_text(pricing))
            if not rows:
                print("  (no rows yet — run a pricing query first)")
            for plan in rows:
                print(
                    f"  • {plan['company_name']}: {plan['plan_name']} - "
                    f"Input Token {format_price(plan['input_tokens'])}, "
                    f"Output Tokens {format_price(plan['output_tokens'])}"
                    f"\n    Source: {plan['source_url'] or 'Unavailable'}"
                    f"\n    Last updated: {plan['scraped_at'] or 'Unavailable'}"
                )

            print("=" * 50)
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")

            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()

    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"

    server_config = config.load_config(config_file)

    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
