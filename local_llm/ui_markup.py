"""The HTML skeleton: header, views, side panels and overlays.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_HEAD = r"""
<!doctype html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>{{APP_NAME}}</title>
 <style>"""

UI_BODY = r"""</style>
</head>
<body>
 <header>
   <div class="brand"><span class="brand-slot">{{APP_LOGO}}</span><strong>{{APP_NAME}}</strong> <span class="build" title="UI build version">build {{UI_BUILD}}</span></div>
   <div id="status">Starting...</div>
   <nav class="views">
     <button id="navChat" class="active" onclick="showView('chat')" data-tip-below data-tip="Talk to the model. Ask questions, paste code or documents, run tools." title="Chat view">Chat</button>
     <button id="navTasks" onclick="showView('tasks')" data-tip-below data-tip="Scheduled or saved jobs the agent runs on demand or on a timer." title="Tasks view">Tasks</button>
     <button id="navHistory" onclick="showView('history')" data-tip-below data-tip="Browse, search, reopen and manage past conversations." title="History view">History</button>
     <button id="navModels" onclick="showView('models')" data-tip-below data-tip="Pick or download a model, and attach a fine-tuned adapter." title="Models view">Models</button>
   </nav>
   <div class="actions">
     <button onclick="newChat()" data-tip-below data-tip="Start a fresh conversation. Clears the current thread from view." title="New chat">New chat</button>
     <button onclick="regenerateLast()" data-tip-below data-tip="Discard the last answer and generate a new one for the same question." title="Regenerate last answer">Regenerate</button>
     <button onclick="showView('history')" data-tip-below data-tip="Browse and search every past conversation." title="Search conversations">Search</button>
     <button onclick="exportChat()" data-tip-below data-tip="Download this conversation as Markdown." title="Export conversation">Export</button>
     <button onclick="togglePrompts()" data-tip-below data-tip="Save and reuse prompts you type often." title="Prompt library">Prompts</button>
     <button id="themeBtn" onclick="toggleTheme()" data-tip-below data-tip="Switch between the dark and light colour scheme." title="Toggle theme">Light</button>
     <button onclick="retrain()" data-tip-below data-tip="Fine-tune the model on your thumbs-up/down feedback so far (LoRA)." title="Retrain on feedback">Retrain</button>
     <button onclick="toggleSettings()" data-tip-below data-tip="Model, tools, generation and memory settings you can change live." title="Open settings">Settings</button>
   </div>
 </header>

 <div id="main">
   <div id="chatView" class="view"><div id="chat"></div></div>

   <div id="tasksView" class="view hidden">
     <div class="task-list">
       <div style="display:flex;justify-content:space-between;align-items:center">
         <strong style="font-size:13px">Tasks</strong>
         <button onclick="toggleTaskForm()">New task</button>
       </div>

       <div id="taskForm" class="form-grid" style="display:none;margin-top:10px">
         <label>Name</label>
         <input id="tfName" type="text" placeholder="Morning news sweep">
         <label>Goal, written as an instruction to the agent</label>
         <textarea id="tfGoal" placeholder="Search for news about MLX released in the last day and summarise anything new in three bullets."></textarea>
         <div class="row">
           <div>
             <label>Repeat every (seconds)</label>
             <input id="tfInterval" type="number" min="0" step="60" value="0">
           </div>
           <div>
             <label data-tip="Tool/reasoning steps per turn before the agent must answer. Big tasks continue past this from a summary.">Max steps</label>
             <input id="tfSteps" type="number" min="1" max="20" value="6">
           </div>
         </div>
         <label>Tools (comma separated, blank for all)</label>
         <input id="tfTools" type="text" placeholder="web_search, fetch_url, remember">
         <label>System prompt override (optional)</label>
         <textarea id="tfSystem" placeholder="Leave blank to use the global system prompt."></textarea>
         <div class="row" style="margin-top:10px">
           <div>
             <label>Keep conversation history</label>
             <select id="tfHistory">
               <option value="false">no, each run is fresh</option>
               <option value="true">yes, runs build on each other</option>
             </select>
           </div>
         </div>
         <div class="row" style="margin-top:12px">
           <button class="primary" onclick="createTask()">Create</button>
           <button onclick="toggleTaskForm()">Cancel</button>
         </div>
         <div class="hint">
           Interval 0 means the task only runs when you press Run. A repeating
           task fires once as soon as it is created, so you find out quickly
           whether the goal is worded well.
         </div>
       </div>

       <div id="taskCards" style="margin-top:12px">loading...</div>
     </div>

     <div class="task-monitor">
       <div id="monitorHeader" style="color:#888;font-size:13px">
         Select a task to watch its runs, or create one.
       </div>
       <div id="runControls" style="display:none;margin-top:10px">
         <div class="row" style="max-width:520px">
           <div>
             <label style="font-size:12px;color:#999">Run</label>
             <select id="runPicker" onchange="openRun(this.value)"></select>
           </div>
           <div style="flex:0 0 auto;display:flex;align-items:flex-end;gap:6px">
             <button onclick="runSelectedTask()">Run now</button>
             <button onclick="cancelSelectedTask()">Cancel</button>
           </div>
         </div>
       </div>
       <div id="runFeed"></div>
     </div>
   </div>

   <div id="historyView" class="view hidden">
     <div class="panel">
       <h3 data-tip="Every conversation stored locally. Reopen one to continue it.">Conversations</h3>
       <div class="row">
         <div><input id="historySearch" type="text" placeholder="search all conversations..." onkeydown="if(event.key==='Enter')runHistorySearch()"></div>
         <div style="flex:0 0 auto"><button onclick="runHistorySearch()" data-tip="Find conversations containing this text." title="Search">Search</button></div>
         <div style="flex:0 0 auto"><button onclick="loadHistory()" data-tip="Show all recent conversations again." title="Show all">Show all</button></div>
       </div>
       <div id="historyList" style="margin-top:12px">loading...</div>
     </div>

     <div class="panel">
       <h3 data-tip="Save or restore everything you have created in this app.">Backup</h3>
       <div class="hint">
         Downloads your conversations, prompts, feedback and indexed documents as
         one JSON file. Restoring merges a backup back in without deleting what is
         already here. Model weights are not included; they are re-downloadable.
       </div>
       <div class="row" style="margin-top:10px">
         <button onclick="downloadBackup()" data-tip="Download everything as a JSON file." title="Download backup">Download backup</button>
         <button onclick="document.getElementById('restoreInput').click()" data-tip="Merge a previously downloaded backup back in." title="Restore backup">Restore backup</button>
       </div>
       <input id="restoreInput" type="file" accept="application/json" style="display:none">
     </div>
   </div>

   <div id="modelsView" class="view hidden">
     <div class="panel">
       <h3>Current</h3>
       <div id="modelCurrent" class="logbox" style="max-height:none">loading...</div>

       <h3>Switch model</h3>
       <table class="models"><tbody id="modelTable"></tbody></table>
       <label style="display:block;font-size:12px;color:#999;margin:12px 0 4px">
         Or any Hugging Face repo id
       </label>
       <div class="row">
         <div><input id="modelCustom" type="text" placeholder="mlx-community/Qwen2.5-7B-Instruct-4bit"></div>
         <div style="flex:0 0 auto"><button onclick="useModel(document.getElementById('modelCustom').value.trim())" data-tip="Queue this model id (Hugging Face / mlx-community) to load." title="Use this model">Use</button></div>
       </div>

       <h3>Adapter and cache</h3>
       <div class="row">
         <div>
           <label style="font-size:12px;color:#999">LoRA adapter</label>
           <select id="adapterSelect"></select>
         </div>
         <div>
           <label style="font-size:12px;color:#999">KV cache cap (0 = unbounded)</label>
           <input id="kvSize" type="number" min="0" step="512">
         </div>
       </div>
       <div class="row" style="margin-top:12px">
         <button class="primary" onclick="applyModel()" data-tip="Load the selected model and adapter. Restarts the model server." title="Apply and restart">Apply and restart</button>
         <button onclick="loadModels()" data-tip="Reload the list of available and cached models." title="Refresh model list">Refresh</button>
       </div>
       <div class="hint">
         Switching to a model that is not cached downloads it on first use, which
         can take minutes and several gigabytes. The log below is the model
         server's own output, including download progress.
       </div>

       <h3>Model server log</h3>
       <div id="modelLog" class="logbox">loading...</div>
     </div>

     <div class="panel">
       <h3 data-tip="How reusable your collected data is for training this or any future model.">Training data</h3>
       <div id="datasetStats" class="logbox" style="max-height:none">loading...</div>
       <div class="hint">
         Your feedback is stored as portable data, not tied to the current model.
         Export it to train this assistant now, or any model later. Approved
         answers become examples to imitate; rejected ("no, wrong") answers and
         your corrections become preference pairs for DPO-style tuning.
       </div>
       <label style="display:block;font-size:12px;color:#999;margin:10px 0 4px">Export format</label>
       <div class="row" style="flex-wrap:wrap;gap:6px">
         <button onclick="exportDataset('chat')" data-tip="Chat messages WITH this assistant's system prompt. Trains a model to be this assistant. Used by the built-in LoRA loop." title="Export chat JSONL">Chat (this assistant)</button>
         <button onclick="exportDataset('bare')" data-tip="Chat messages with NO system prompt. Model-neutral: train a different base model or persona." title="Export bare JSONL">Bare Q&amp;A (any model)</button>
         <button onclick="exportDataset('preference')" data-tip="{prompt, chosen, rejected} pairs from corrections and good/bad answers. For DPO-style preference tuning later." title="Export preference pairs">Preference pairs (DPO)</button>
         <button onclick="exportDataset('raw')" data-tip="Every column as JSONL. A lossless archive you can reshape into any format in the future." title="Export raw archive">Raw archive</button>
       </div>
       <div class="row" style="margin-top:8px">
         <label class="agent-toggle" data-tip="Only export rows you have marked reviewed. Curated data trains better on any model."><input id="exportReviewedOnly" type="checkbox"> reviewed only</label>
         <button onclick="loadDatasetStats()" data-tip="Refresh the dataset counts." title="Refresh stats">Refresh</button>
       </div>
     </div>
     <div class="panel">
       <h3 data-tip="Index your own documents so answers come from your material, with sources.">Knowledge base</h3>
       <div id="docsStats" class="logbox" style="max-height:none">loading...</div>
       <div class="hint">
         Indexed documents are searched on every question and the best passages are
         added to the prompt with their source path. Uses SQLite full-text ranking,
         so it needs no embedding model and no extra memory. PDFs, docx, notebooks,
         spreadsheets and text are all read.
       </div>
       <div id="dropZone" data-tip="Drag files here, or click to choose, to add them to the knowledge base.">
         <strong>Drop files here</strong> or click to choose &mdash; PDFs, Word docs, spreadsheets, notebooks, text.
       </div>
       <input id="fileInput" type="file" multiple style="display:none">
       <label style="display:block;font-size:12px;color:#999;margin:10px 0 4px">File or folder to index (relative to the project)</label>
       <div class="row">
         <div><input id="docsPath" type="text" placeholder="docs"></div>
         <div style="flex:0 0 auto"><button onclick="indexDocs()" data-tip="Read and index this file or folder into the knowledge base." title="Index">Index</button></div>
       </div>
       <div class="row" style="margin-top:8px">
         <div><input id="docsUrl" type="text" placeholder="https://example.com/docs"></div>
         <div style="flex:0 0 auto"><button onclick="indexUrl()" data-tip="Fetch a web page and add it to the knowledge base so you can ask about it later without re-fetching." title="Index URL">Index URL</button></div>
       </div>
       <div class="row" style="margin-top:8px">
         <div><input id="docsQuery" type="text" placeholder="test a search..."></div>
         <div style="flex:0 0 auto"><button onclick="searchDocs()" data-tip="Preview what the model would retrieve for this question." title="Search">Search</button></div>
         <div style="flex:0 0 auto"><button onclick="clearDocs()" data-tip="Remove every indexed document. Your files are not touched." title="Clear">Clear</button></div>
       </div>
       <label style="display:block;font-size:12px;color:#999;margin:12px 0 4px" data-tip="Limit retrieval to specific documents, so answers come only from what you choose.">Answer only from these documents</label>
       <div id="docScope" style="max-height:150px;overflow-y:auto"></div>
       <div class="row" style="margin-top:8px">
         <button onclick="applyScope()" data-tip="Restrict retrieval to the ticked documents." title="Apply scope">Apply scope</button>
         <button onclick="clearScope()" data-tip="Search the whole knowledge base again." title="Use all documents">Use all</button>
       </div>
       <pre id="docsResult" class="gbody gdiff" style="display:none;margin-top:8px"></pre>
     </div>
     <div class="panel">
       <h3 data-tip="The local codebase the agent reads and edits in place. Review here before you push.">Codebase</h3>
       <label style="display:block;font-size:12px;color:#999;margin:4px 0 4px">Project directory (PROJECT_DIR)</label>
       <div class="row">
         <div><input id="cfgProjectDir" type="text" placeholder="/Users/you/path/to/repo"></div>
         <div style="flex:0 0 auto"><button onclick="openBrowser()" data-tip="Browse your folders and pick the project directory." title="Browse for a folder">Browse...</button></div>
         <div style="flex:0 0 auto"><button onclick="saveProjectDir()" data-tip="Point the file tools at this local codebase. Empty = sandbox workspace only." title="Set project directory">Set</button></div>
       </div>
       <div class="hint">
         When set, the agent reads and edits files in this directory in place. It
         never touches .git, and it can't write outside the directory. Changes are
         yours to review with git and push manually — nothing is committed for you.
       </div>
       <div id="projectStatus" class="logbox" style="max-height:none;margin-top:10px">loading...</div>
       <div class="row" style="margin-top:8px">
         <button onclick="loadProjectStatus()" data-tip="Refresh git status and the list of files changed this session." title="Refresh">Refresh</button>
         <button onclick="loadProjectDiff()" data-tip="Show the uncommitted git diff so you can review before pushing." title="Show diff">Show diff</button>
         <button onclick="revertChanges()" data-tip="Undo this session's edits: restore modified files and delete files created this session. Cannot be undone." title="Revert session changes">Revert session changes</button>
       </div>
       <div class="hint" id="sandboxHint" style="margin-top:8px"></div>
       <pre id="projectDiff" class="gbody gdiff" style="display:none;margin-top:8px"></pre>
     </div>
   </div>

   <div id="browser" class="hidden">
     <div id="browserBox">
       <div id="browserPath">~</div>
       <div id="browserList"></div>
       <div class="row" style="padding:10px 14px;border-top:1px solid var(--border)">
         <button onclick="chooseCurrentFolder()" data-tip="Use the folder shown above as the project directory." title="Use this folder">Use this folder</button>
         <button onclick="closeBrowser()" title="Cancel">Cancel</button>
       </div>
     </div>
   </div>

   <div id="palette" class="hidden">
     <div id="paletteBox">
       <input id="paletteInput" type="text" placeholder="Type a command..." autocomplete="off">
       <div id="paletteList"></div>
     </div>
   </div>

   <aside id="promptsPanel" class="hidden">
     <h3>Prompt library</h3>
     <div class="hint">Save prompts you type often, then insert one into the message box with a click.</div>
     <div id="promptList" style="margin-top:10px">loading...</div>
     <label style="display:block;font-size:12px;color:#999;margin:12px 0 4px">Save the current message box as</label>
     <div class="row">
       <div><input id="promptName" type="text" placeholder="name, e.g. code-review"></div>
       <div style="flex:0 0 auto"><button onclick="savePrompt()" data-tip="Save whatever is in the message box under this name." title="Save prompt">Save</button></div>
     </div>
     <div class="row" style="margin-top:10px">
       <button onclick="togglePrompts()" title="Close">Close</button>
     </div>
   </aside>

   <aside id="settings">
     <h3>Generation</h3>
     <label>System prompt</label>
     <textarea id="cfgSystem"></textarea>
     <div class="row">
       <div>
         <label data-tip="Longest reply the model may generate, in tokens. Higher = longer answers but more memory and time.">Max tokens</label>
         <input id="cfgMaxTokens" type="number" min="16" max="32768" step="16">
       </div>
       <div>
         <label data-tip="Randomness of replies. 0 = deterministic and focused; higher = more varied and creative.">Temperature</label>
         <input id="cfgTemperature" type="number" min="0" max="2" step="0.05">
       </div>
     </div>
     <div class="row">
       <div>
         <label data-tip="Total working window (prompt + reply). Auto-sized to your RAM; larger holds more history but uses more memory. Above ~60% of this, large prompts are chunked.">Context size (tokens)</label>
         <input id="cfgContext" type="number" min="512" max="131072" step="512">
       </div>
       <div>
         <label data-tip="How many past messages to carry into each request. Fewer = less memory, less continuity.">History turns</label>
         <input id="cfgHistory" type="number" min="0" max="200" step="1">
       </div>
     </div>
     <div class="meter"><div id="ctxBar"></div></div>
     <div class="hint" id="ctxHint">Context usage in this conversation.</div>

     <h3 style="margin-top:18px">Agent</h3>
     <div class="row">
       <div>
         <label data-tip="Whether new chats start in agent mode (tools + step reasoning) or as plain single replies.">Enabled by default</label>
         <select id="cfgAgent">
           <option value="false">off</option>
           <option value="true">on</option>
         </select>
       </div>
       <div>
         <label>Max steps</label>
         <input id="cfgAgentSteps" type="number" min="1" max="20" step="1">
       </div>
     </div>
     <div class="row">
       <div>
         <label data-tip="Which web search provider the search tool uses. ddg needs no key; brave/tavily/searxng may need one.">Search backend</label>
         <select id="cfgSearchBackend">
           <option value="ddg">ddg</option>
           <option value="brave">brave</option>
           <option value="tavily">tavily</option>
           <option value="searxng">searxng</option>
         </select>
       </div>
       <div>
         <label data-tip="How many results the search tool returns per query.">Search results</label>
         <input id="cfgSearchResults" type="number" min="1" max="10" step="1">
       </div>
     </div>
     <label data-tip="Max characters a tool result may contribute before it is summarised or truncated.">Tool result limit (chars)</label>
     <input id="cfgToolChars" type="number" min="200" max="40000" step="200">

     <div style="margin-top:14px" class="row">
       <button class="primary" onclick="saveConfig()" data-tip="Save these settings. Most apply immediately, no restart needed." title="Apply settings">Apply</button>
       <button onclick="restartModel()" data-tip="Restart the local model server. Use if it becomes unresponsive." title="Restart model server">Restart model</button>
     </div>
     <div class="hint">
       Context size takes effect on the next message. The KV cache size passed to
       the model server only changes on restart.
     </div>

     <h3 style="margin-top:18px">Performance</h3>
     <div class="row">
       <div>
         <label data-tip="How much of a tool result is fed back to the model as context.">Tool result into context (chars)</label>
         <input id="cfgToolChars2" type="number" min="200" max="40000" step="100">
       </div>
       <div>
         <label data-tip="Temperature used only when the model is choosing a tool. 0 keeps tool-calls deterministic.">Tool step temperature</label>
         <input id="cfgToolTemp" type="number" min="0" max="2" step="0.05">
       </div>
     </div>
     <label class="agent-toggle" style="margin-top:8px" data-tip="Skip the model's hidden &lt;think&gt; phase. Faster replies; may reduce reasoning quality on hard questions.">
       <input id="cfgThinking" type="checkbox"> disable thinking mode
     </label>
     <label class="agent-toggle" data-tip="Route obvious requests (a URL, arithmetic) directly to a tool without a model call. Faster and cheaper.">
       <input id="cfgFastPath" type="checkbox"> deterministic fast path
     </label>
     <label class="agent-toggle" data-tip="Keep the prompt prefix stable between steps so the model server can reuse its cache. Faster, uses a bit more memory.">
       <input id="cfgStablePrefix" type="checkbox"> stable prefix (cache friendly)
     </label>
     <label class="agent-toggle" data-tip="Condense oversized tool results before feeding them back, to save context on smaller machines.">
       <input id="cfgSummarise" type="checkbox"> summarise long tool results
     </label>
     <div class="row" style="margin-top:10px">
       <button onclick="saveConfig()">Apply</button>
       <button onclick="loadPerf()">Refresh stats</button>
     </div>
     <div class="tools-list" id="perfStats" style="margin-top:8px">no samples yet</div>
     <div id="prefillCurve" style="margin-top:10px"></div>
     <div class="hint">
       Prompt tokens climbing step over step within one run is re-prefill cost.
       Flat means the prefix is being reused.
     </div>
     <div class="hint" id="prefixWarn" style="display:none"></div>

     <h3 style="margin-top:18px">Memory</h3>
     <div class="row">
       <div><input id="memKey" type="text" placeholder="key"></div>
       <div><input id="memValue" type="text" placeholder="value"></div>
     </div>
     <div class="row" style="margin-top:8px">
       <button onclick="saveMemory()">Store</button>
       <button onclick="loadMemory()">Refresh</button>
     </div>
     <div class="tools-list" id="memoryList" style="margin-top:8px">loading...</div>
     <div class="hint">
       Notes the agent stores with the remember tool, and anything you add here.
       They persist across restarts.
     </div>

     <h3 style="margin-top:18px">Tools</h3>
     <div class="tools-list" id="toolsList">loading...</div>
   </aside>
 </div>

 <footer id="composer">
   <textarea id="input" placeholder="Send a message. Shift+Enter for a new line." rows="1" data-tip="Type here. Enter sends, Shift+Enter adds a line. Paste large files freely; they are processed in chunks." title="Message input"></textarea>
   <label class="agent-toggle" data-tip="Agent mode lets the model call tools (search, fetch, weather, calculator) and reason in steps. Off = a plain single reply." title="Toggle agent mode"><input id="agentToggle" type="checkbox"> agent</label>
   <button id="stopBtn" onclick="stopStream()" disabled data-tip="Stop the current response. Keeps whatever streamed so far." title="Stop generating">Stop</button>
   <button id="sendBtn" class="primary" onclick="send()" data-tip="Send your message (or press Enter)." title="Send message">Send</button>
 </footer>

 <script>"""

UI_TAIL = r"""</script>
</body>
</html>
"""

__all__ = ["UI_HEAD", "UI_BODY", "UI_TAIL"]
