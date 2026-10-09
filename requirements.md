# PriceScout Requirements

## Goal

LLM providers ki pricing ko automatically collect karke SQLite mein save karna,
aur user ke natural-language questions ka jawab saved data se dena.

## Core Workflow

1. **Scheduled collection:** System har din configured competitor websites ki
   pricing pages scrape kare.
2. **Data storage:** Scraped pricing, model name, input/output rates, source URL,
   aur collection timestamp SQLite mein save/update hon.
3. **Chatbot:** User query kare, to Claude/MCP stored database data ko read karke
   concise answer ya provider/model comparison de.
4. **On-demand refresh:** Agar requested provider/model ka data missing ya stale ho,
   chatbot scraper ko trigger karke latest data save kar sake.

## Initial Scope

- Providers: DeepInfra, Groq, Fireworks AI, CloudRift AI
- Scraping: Python MCP server + Firecrawl
- Orchestration: Anthropic Claude client
- Database: SQLite MCP server
- Local files: Filesystem MCP server
- Interface: Command-line chatbot
- Refresh frequency: Daily, configurable

## Important Behavior

- Chatbot normally database se jawab de; har query par website scrape na kare.
- Source URL aur last-updated time answer mein available hon.
- Different pricing units (per-token vs GPU/hour) ko clearly distinguish kiya jaye.
- Scraping/API/database errors user ko clearly report hon.

## Success Criteria

- Daily job configured websites se latest data collect kare.
- Pricing records duplicate hone ke bajaye update/upsert hon.
- User provider, model, price, aur comparison queries pooch sake.
- Answers saved source data par based hon aur source/timestamp show karein.
