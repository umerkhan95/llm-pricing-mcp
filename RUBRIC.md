# Project Rubric — LLM Pricing MCP

> Verbatim rubric text as supplied by the user on 2026-09-14 (recovered from session
> 7c75b6fb on 2026-09-23). This is the assessment criteria for the project in this
> folder: `starter_server.py` (scraper MCP server) + `starter_client.py` (client).
> Code shapes quoted below are **required by the rubric** — do not refactor them away.

Project Rubric
On the following pages, you'll find the rubric that will be used to assess your project, please check the rubric while completing the project, and before submission.

Project Starter Files
If working locally, you can download the starter files from the workspace on the following page. Just select the files and right click, choose "download file".

To complete the project, your scraper server needs to talk to the Firecrawl service. This requires a free API key.

Go to the Sign-up Page Use this link to create a new, free account: (opens in a new tab)https://www.firecrawl.dev/signin(opens in a new tab) You can sign up quickly using a Google, GitHub, or email account.
Find Your API Key Once you have signed in, you will be on your dashboard.
On the left-hand menu, click on the "API Keys" section.
You will see your first API key already created for you, labeled "Default".
Copy Your Key Copy the API key (it will be a long string that starts with fc-...).
Add the Key to Your Project Open the .env file in your project folder. Paste your Firecrawl API key directly after FIRECRAWL_API_KEY=. Your .env file should look like this (but with your real keys):
FIRECRAWL_API_KEY=fc-your-firecrawl-key-goes-here
That's it! Once your key is in the .env file, your starter_server.py will be able to connect to Firecrawl and scrape the websites.

Project Instructions
Here are the step-by-step instructions to guide you through completing this project. We'll build the server's tools first, then wire up the client to use them.

Step 0: Prerequisites (Your Environment)
Before you write any code, let's make sure your environment is ready.

Set API Keys: Create a new file in the root of your project folder named .env. This file is used to store your secret API keys. Open it and add your keys from Anthropic and Firecrawl:
ANTHROPIC_API_KEY=your-anthropic-key-goes-here
FIRECRAWL_API_KEY=your-firecrawl-key-goes-here
Configure Your Server Path: Open the server_config.json file. We need to tell the client how to find and run your starter_server.py file. Find the "llm_inference" server configuration:
"llm_inference": {
"command": "uv",
"args": ["run", "mcp-project/server.py"]
},
Change the "args" to point to the starter_server.py file you will be editing:

"llm_inference": {
 "command": "uv",
 "args": ["run", "starter_server.py"]
},
Step 1: Build Your Scraper Server (starter_server.py)
Open starter_server.py. This file will become your custom MCP server that scrapes websites. We need to complete the two functions marked as tools.

Task 1.1: Complete the scrape_websites tool
This function will use Firecrawl to scrape a list of websites, save their content to files, and update a central scraped_metadata.json file.

Inside the scrape_websites function, replace the # continue your solution here ... comment with logic to do the following:

Load Existing Metadata: Try to open and load scraped_metadata.json. If it doesn't exist or is empty, initialize scraped_metadata as an empty dictionary {}.
Initialize: Create a list called successful_scrapes to hold the names of providers we successfully scrape.
Loop Through Websites: Iterate over the websites dictionary (which contains provider_name -> url pairs).
Inside the Loop (for each website):
Wrap the scraping logic in a try...except block to handle any errors.
Log that you are starting to scrape (e.g., logger.info(f"Scraping {provider_name}: {url}")).
Call Firecrawl: scrape_result = app.scrape(url, formats=formats).model_dump().
Create a metadata dictionary for this provider, including its name, URL, domain, and scrape time.
If Successful: Check if scrape_result.get('success', False) is true.
Save Content: Loop through the formats (e.g., 'markdown', 'html'). For each format, get the content from scrape_result, create a filename (like f"{provider_name}_{format_type}.txt"), and write the content to a file in the SCRAPE_DIR.
Update Metadata: Add the filenames to your metadata dictionary (under a content_files key). Add the page title and description from the scrape result.
Add the provider_name to your successful_scrapes list.
If Failed: Log the error from scrape_result.
Finally: Add your metadata dictionary to the main scraped_metadata dictionary (e.g., scraped_metadata[provider_name] = metadata).
After the Loop:
Write the entire scraped_metadata dictionary back to the scraped_metadata.json file.
Log the final results.
Return the successful_scrapes list.
Task 1.2: Complete the extract_scraped_info tool
This function will read the data you just saved.

Inside the extract_scraped_info function, replace the # contine your response here ... comment with logic to do the following:

Load Metadata: Open and load the scraped_metadata.json file inside a try...except block.
Find a Match: Loop through the scraped_metadata dictionary. For each provider_name and metadata entry, check if your identifier matches any of these:
The provider_name
The metadata.get('url', '')
The metadata.get('domain', '')
If a Match is Found:
Make a copy of the metadata (e.g., result = metadata.copy()).
Check if the metadata has 'content_files'. If it does, create a new result['content'] dictionary.
Loop through the content_files (e.g., format_type, filename).
Read the content from the file (e.g., os.path.join(SCRAPE_DIR, filename)) and store it in result['content'][format_type].
Return the result dictionary as a formatted JSON string: return json.dumps(result, indent=2).
If No Match/Error: If the loop finishes with no match or if you hit an error (like FileNotFoundError), return a string message like f"There's no saved information related to identifier '{identifier}'.".
Step 2: Build Your Client Application (starter_client.py)
Open starter_client.py. This is your "manager" app that will talk to your server. We just need to fill in the sections marked # complete.

In Configuration.load_config****:
Implement the logic to open and read the file_path.
Load the JSON: config = json.load(f).
Add error handling (the try...except block is provided).
Check if mcpServers is in the config: if "mcpServers" not in config: ...
Return the config.
In Server.initialize****:
The server_params object is empty. Fill it using the self.config dictionary:
server_params = StdioServerParameters(
command=command,
args=self.config["args"],
env={**os.environ,** self.config["env"]} if self.config.get("env") else None,
)
In Server.list_tools****:
Check if the session exists: if not self.session: ...
Call the session to get tools: tools_response = await self.session.list_tools().
Format the response. The starter code provides the loop; you just need to populate the tool_def dictionary:
tool_def: ToolDefinition = {
"name": tool.name,
"description": tool.description,
"input_schema": tool.inputSchema
}
In Server.execute_tool****:
Implement the retry loop. The core of the try block is:
logging.info(f"Executing {tool_name}...")
result = await self.session.call_tool(name=tool_name, arguments=arguments, read_timeout_seconds=timedelta(seconds=60))
return result
In DataExtractor.extract_and_store_data****:
After loading the pricing_data, loop through the plans.
Inside the for plan in pricing_data.get("plans", []): loop, add the sqlite_server.execute_tool call to insert the data into the database.
await self.sqlite_server.execute_tool("write_query", {
"query": f"""
INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
VALUES (
'{pricing_data.get("company_name", "Unknown")}',
'{plan.get("plan_name", "Unknown Plan")}',
'{plan.get("input_tokens", 0)}',
'{plan.get("output_tokens", 0)}',
'{plan.get("currency", "USD")}',
'{plan.get("billing_period", "unknown")}',
'{json.dumps(plan.get("features", []))}',
'{plan.get("limitations", "")}',
'{user_query}')
"""
})
In ChatSession.process_query****:
Model Name: Change <ENTER_MODEL_NAME> to a Vocareum-available model. Use the same dated model the solution uses: 'claude-sonnet-4-5-20250929'.
if content.type == 'text':***** block:*
Add the text to the full_response: full_response += content.text + "\n".
Add the content to the assistant's message: assistant_content.append(content).
Check if this is the only content. If so, the model is done:
if len(response.content) == 1:
process_query = False
elif content.type == 'tool_use':***** block:*
This is the full tool-use loop. The logic is:
Append the tool use request to assistant_content and then to messages.
Get the tool id, args, and name from the content.
Find the server that has this tool.
execute_tool on that server.
Append the tool_result to messages.
Call self.anthropic.messages.create again with the new messages list.
Check if the new response is just text, and if so, stop the loop.
In ChatSession.show_stored_data****:
Add the tool call to read from the pricing_plans table:
pricing = await self.sqlite_server.execute_tool("read_query", {
    "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
})

print("\nRecently Stored Data:")
print("=" * 50)

print("\nPricing Plans:")
# result.content[0].text is a JSON string of rows — parse it first.
import json
rows = json.loads(pricing.content[0]["text"])
for plan in rows:
    print(f"  • {plan['company_name']}: {plan['plan_name']} - Input Token ${plan['input_tokens']}, Output Tokens ${plan['output_tokens']}")

print("=" * 50)

### Step 3: Run and Test Your Project

You are now ready to run your application!

1. **Run the Client:** In your terminal (with your virtual environment activated), run:
```bash
python starter_client.py
Test Your Tools: Your client will start, connect to all 3 servers (your scraper, sqlite, and filesystem), and give you a Query: prompt.
Use the Test Prompts from the README.md:
First, scrape the sites:
"scrape these sites: {'cloudrift': 'https://www.cloudrift.ai/inference', 'deepinfra': 'https://deepinfra.com/pricing', 'fireworks': 'https://fireworks.ai/pricing#serverless-pricing', 'groq': 'https://groq.com/pricing'}"
Then, ask questions about the scraped content:
"How much does cloudrift ai (https://www.cloudrift.ai/inference) charge for deepseek v3?"
"How much does deepinfra (https://deepinfra.com/pricing) charge for deepseek v3"
"Compare cloudrift ai and deepinfra's costs for deepseek v3"
Check your database:
Type show data to see the structured data your DataExtractor saved to the SQLite database.

---

## Grader note carried over from the session

> One suggestion would be to read the data from the SQLite database when comparing the
> costs of Cloudrift AI and DeepInfra, and then run the comparison prompt on the
> retrieved data. Otherwise the scrape runs again, consuming extra API credits.

---

# Grading Rubric (criteria table)

> Supplied by the user on 2026-09-23. This is what the project is actually
> assessed against; the instructions above are the how-to.

## Custom MCP Server

| Criteria | Submission Requirements |
|---|---|
| Implement `scrape_websites` tool to scrape, persist, and track metadata | Running the tool creates `scraped_content/` with one file per `{provider}_{format}.txt` for each successful provider and creates/updates `scraped_metadata.json` containing provider name, url, domain, scraped_at, content_files, title, description. Tool returns a list whose length equals number of successful scrapes. |
| Implement `extract_scraped_info` tool to load metadata and return file contents | Calling the tool with an identifier that matches provider name, URL, or domain returns a formatted JSON string whose **`content` field includes loaded text for each available format**; non-matching identifier returns a plain-text message indicating no saved information. |

## AI-Powered MCP Client

| Criteria | Submission Requirements |
|---|---|
| List available MCP tools | `Server.list_tools` checks session existence, calls `list_tools()`, and returns a list of dicts containing name, description, input_schema. |
| Execute tools with retries | `Server.execute_tool` contains a retry loop and within each attempt logs execution, calls `session.call_tool(...)` with a 60-second read timeout, and returns the result on success. |
| Persist structured pricing to SQLite via MCP | `DataExtractor.extract_and_store_data` iterates `pricing_data["plans"]` and executes the provided `write_query` insertion **exactly as specified (including column order and `json.dumps` for features)**. |
| Drive tool use via LLM and complete tool-use loop | `ChatSession.process_query` sets a real model name (e.g. `claude-3-5-sonnet-20240620`), appends text chunks to `full_response`, appends assistant content, handles single-text completion exit, and for `tool_use` performs the full 1–7 loop (append tool request, locate server, execute tool, append tool result, call model again, and exit if next response is text only). |
| Display recently stored data from DB | `ChatSession.show_stored_data` executes the provided `read_query`, prints the header lines, iterates result rows, and prints the formatted bullet lines including company, plan, and token pricing, then a closing separator. |

## Orchestrate an Agentic Workflow

| Criteria | Submission Requirements |
|---|---|
| Orchestrate a functional multi-agent MCP workflow | The running client connects to three MCP servers (custom scraper, SQLite, filesystem where applicable) and, from natural-language queries, delegates to the correct tools to scrape, analyze, and store competitor pricing, then answers follow-ups using stored data. |
| Demonstrate comparison Q&A | Screenshot showing `Compare cloudrift ai and deepinfra's costs for deepseek v3` and a full natural-language answer. |
| Demonstrate scraping success | Screenshot showing the `scrape these sites: {...}` query and output confirming success count (e.g. "Successfully scraped 4 out of 4 websites"). |
