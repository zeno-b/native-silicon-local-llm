"""Front-end: the tasks view (scheduling, runs, live feed).

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_JS_TASKS = r"""   // ------------------------------------------------------------- tasks ---

   function toggleTaskForm() {
     var form = document.getElementById("taskForm");
     form.style.display = form.style.display === "none" ? "block" : "none";
   }

   function statusPill(status) {
     var pill = document.createElement("span");
     pill.className = "pill " + (status || "");
     pill.textContent = status || "never run";
     return pill;
   }

   function relativeTime(value) {
     if (!value) return "never";
     var then = new Date(value.indexOf("Z") < 0 && value.indexOf("+") < 0 ? value + "Z" : value);
     var seconds = Math.round((Date.now() - then.getTime()) / 1000);
     if (isNaN(seconds)) return value;
     if (seconds < 0) return "in " + Math.abs(seconds) + "s";
     if (seconds < 60) return seconds + "s ago";
     if (seconds < 3600) return Math.round(seconds / 60) + "m ago";
     if (seconds < 86400) return Math.round(seconds / 3600) + "h ago";
     return Math.round(seconds / 86400) + "d ago";
   }

   function renderPrefillCurve(steps) {
     var box = document.getElementById("prefillCurve");
     box.innerHTML = "";
     if (!steps || steps.length < 2) return;

     var values = steps.map(function(s) { return s.prompt_tokens || 0; });
     var peak = Math.max.apply(null, values) || 1;
     var total = values.reduce(function(a, b) { return a + b; }, 0);

     var title = document.createElement("div");
     title.style.fontSize = "11px";
     title.style.opacity = "0.75";
     title.style.marginBottom = "4px";
     title.textContent = "prompt tokens per step (" + total + " total prefill)";
     box.appendChild(title);

     steps.forEach(function(s, i) {
       var row = document.createElement("div");
       row.style.display = "flex";
       row.style.alignItems = "center";
       row.style.gap = "6px";
       row.style.fontSize = "11px";
       row.style.lineHeight = "1.5";

       var label = document.createElement("span");
       label.style.opacity = "0.6";
       label.style.minWidth = "18px";
       label.textContent = "s" + (s.step === null ? i + 1 : s.step);

       var track = document.createElement("span");
       track.style.flex = "1";
       track.style.height = "9px";
       track.style.borderRadius = "3px";
       track.style.background = "rgba(127,127,127,0.18)";
       track.style.overflow = "hidden";

       var fill = document.createElement("span");
       fill.style.display = "block";
       fill.style.height = "100%";
       fill.style.width = Math.round(((s.prompt_tokens || 0) / peak) * 100) + "%";
       // Growth across steps is the thing to notice, so colour by it.
       var growing = i > 0 && (s.prompt_tokens || 0) > values[i - 1] * 1.15;
       fill.style.background = growing ? "#c2703a" : "#5a8f6f";
       track.appendChild(fill);

       var value = document.createElement("span");
       value.style.opacity = "0.7";
       value.style.minWidth = "40px";
       value.style.textAlign = "right";
       value.textContent = String(s.prompt_tokens || 0);

       row.appendChild(label);
       row.appendChild(track);
       row.appendChild(value);
       box.appendChild(row);
     });

     var verdict = document.createElement("div");
     verdict.style.fontSize = "11px";
     verdict.style.marginTop = "5px";
     var growth = values[values.length - 1] / (values[0] || 1);
     if (growth > 1.5) {
       verdict.style.color = "#c2703a";
       verdict.textContent = "Prompt grew " + growth.toFixed(1) +
         "x across the run. Shrink tool results or the tool list.";
     } else {
       verdict.style.opacity = "0.7";
       verdict.textContent = "Prompt stayed flat across steps.";
     }
     box.appendChild(verdict);
   }

   async function loadRunCurve(convId) {
     if (!convId) return;
     try {
       var out = await fetchJSON("/api/metrics/run/" + encodeURIComponent(convId));
       renderPrefillCurve(out.data.steps || []);
     } catch (err) {
       /* the run had no agent steps; nothing to draw */
     }
   }

   async function loadPerf() {
     var box = document.getElementById("perfStats");
     try {
       var out = await fetchJSON("/api/metrics/summary");
       var chat = out.data.chat || {};
       var step = out.data.agent_step || {};
       var lines = [];
       if (chat.samples) {
         lines.push("chat: " + chat.samples + " samples, " +
                    Math.round(chat.avg_duration_ms) + "ms avg");
       }
       if (step.samples) {
         lines.push("agent steps: " + step.samples);
         lines.push("  prompt tokens avg " + Math.round(step.avg_prompt_tokens || 0));
         lines.push("  ttft avg " + Math.round(step.avg_ttft_ms || 0) + "ms");
         lines.push("  decode " + (step.avg_decode_tps || 0).toFixed(1) + " tok/s");
         lines.push("  total prefill " + (step.total_prompt_tokens || 0) + " tokens");
       }
       box.textContent = lines.join("\n") || "no samples yet";
     } catch (err) {
       box.textContent = "stats unavailable: " + err.message;
     }
   }

   async function loadTasks() {
     try {
       var out = await fetchJSON("/api/tasks");
       renderTasks(out.data.tasks || []);
     } catch (err) {
       document.getElementById("taskCards").textContent = "Could not load tasks: " + err.message;
     }
   }

   function renderTasks(items) {
     var host = document.getElementById("taskCards");
     host.innerHTML = "";
     if (!items.length) {
       host.textContent = "No tasks yet.";
       return;
     }
     items.forEach(function(task) {
       var card = document.createElement("div");
       card.className = "card" + (selectedTask === task.id ? " selected" : "");
       card.onclick = function(e) {
         if (e.target.tagName === "BUTTON") return;
         selectTask(task.id);
       };

       var head = document.createElement("h4");
       var title = document.createElement("span");
       title.textContent = task.name;
       head.appendChild(title);
       head.appendChild(statusPill(task.live ? "running" : task.last_status));
       card.appendChild(head);

       var goal = document.createElement("div");
       goal.className = "goal";
       goal.textContent = task.goal;
       card.appendChild(goal);

       var meta = document.createElement("div");
       meta.className = "meta";
       var parts = [];
       parts.push(task.enabled ? "enabled" : "disabled");
       parts.push(task.interval_seconds > 0 ? "every " + task.interval_seconds + "s" : "manual");
       parts.push(task.run_count + " runs");
       parts.push("last " + relativeTime(task.last_run_at));
       if (task.live) {
         parts.push("step " + task.live.step + "/" + task.live.max_steps);
         if (task.live.tool) parts.push("tool " + task.live.tool);
       } else if (task.next_run_at) {
         parts.push("next " + relativeTime(task.next_run_at));
       }
       meta.textContent = parts.join(" | ");
       card.appendChild(meta);

       var actions = document.createElement("div");
       actions.className = "card-actions";
       actions.appendChild(taskButton("Run", function() { runTask(task.id); }));
       actions.appendChild(taskButton("Cancel", function() { cancelTask(task.id); }));
       actions.appendChild(taskButton(task.enabled ? "Disable" : "Enable", function() {
         updateTask(task.id, { enabled: !task.enabled });
       }));
       actions.appendChild(taskButton("Delete", function() { deleteTask(task.id, task.name); }));
       card.appendChild(actions);

       host.appendChild(card);
     });
   }

   function taskButton(label, handler) {
     var button = document.createElement("button");
     button.textContent = label;
     button.onclick = handler;
     return button;
   }

   async function createTask() {
     var body = {
       name: document.getElementById("tfName").value.trim(),
       goal: document.getElementById("tfGoal").value.trim(),
       interval_seconds: Number(document.getElementById("tfInterval").value) || 0,
       max_steps: Number(document.getElementById("tfSteps").value) || 6,
       tools: document.getElementById("tfTools").value.trim(),
       system_prompt: document.getElementById("tfSystem").value.trim() || null,
       use_history: document.getElementById("tfHistory").value === "true",
       enabled: true
     };
     if (!body.name || !body.goal) {
       alert("A task needs a name and a goal.");
       return;
     }
     try {
       var out = await fetchJSON("/api/tasks", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       document.getElementById("tfName").value = "";
       document.getElementById("tfGoal").value = "";
       toggleTaskForm();
       selectTask(out.data.task.id);
       loadTasks();
     } catch (err) {
       alert("Could not create the task: " + err.message);
     }
   }

   async function updateTask(taskId, patch) {
     try {
       await fetchJSON("/api/tasks/" + taskId, {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(patch)
       });
       loadTasks();
     } catch (err) {
       alert("Could not update the task: " + err.message);
     }
   }

   async function deleteTask(taskId, name) {
     if (!window.confirm("Delete " + name + " and its run history?")) return;
     try {
       await fetchJSON("/api/tasks/" + taskId, { method: "DELETE" });
       if (selectedTask === taskId) {
         selectedTask = null;
         closeRun();
         document.getElementById("runControls").style.display = "none";
         document.getElementById("monitorHeader").textContent = "Select a task to watch its runs.";
         document.getElementById("runFeed").innerHTML = "";
       }
       loadTasks();
     } catch (err) {
       alert("Could not delete the task: " + err.message);
     }
   }

   async function runTask(taskId) {
     try {
       var out = await fetchJSON("/api/tasks/" + taskId + "/run", { method: "POST" });
       if (out.data.error) {
         alert(out.data.error);
         return;
       }
       selectedTask = taskId;
       await loadRuns(taskId);
       openRun(out.data.run_id);
       loadTasks();
     } catch (err) {
       alert("Could not start the run: " + err.message);
     }
   }

   async function cancelTask(taskId) {
     try {
       await fetchJSON("/api/tasks/" + taskId + "/cancel", { method: "POST" });
       loadTasks();
     } catch (err) {
       alert("Could not cancel: " + err.message);
     }
   }

   function runSelectedTask() { if (selectedTask) runTask(selectedTask); }
   function cancelSelectedTask() { if (selectedTask) cancelTask(selectedTask); }

   async function selectTask(taskId) {
     selectedTask = taskId;
     document.getElementById("runControls").style.display = "block";
     try {
       var out = await fetchJSON("/api/tasks/" + taskId);
       var task = out.data.task;
       document.getElementById("monitorHeader").textContent =
         task.name + " -- " + task.goal;
       renderRunPicker(out.data.runs || []);
       if (out.data.runs && out.data.runs.length) {
         openRun(out.data.runs[0].id);
       } else {
         closeRun();
         document.getElementById("runFeed").innerHTML = "";
       }
       loadTasks();
     } catch (err) {
       document.getElementById("monitorHeader").textContent = "Could not load task: " + err.message;
     }
   }

   async function loadRuns(taskId) {
     try {
       var out = await fetchJSON("/api/tasks/" + taskId + "/runs?limit=20");
       renderRunPicker(out.data.runs || []);
     } catch (err) {
       // Leave the picker as it is; the stream is the important part.
     }
   }

   function renderRunPicker(runs) {
     var picker = document.getElementById("runPicker");
     picker.innerHTML = "";
     runs.forEach(function(run) {
       var option = document.createElement("option");
       option.value = run.id;
       option.textContent = run.status + " -- " + relativeTime(run.started_at) +
                            " -- " + run.trigger;
       picker.appendChild(option);
     });
     if (currentRunId) picker.value = currentRunId;
   }

   function closeRun() {
     if (runSource) {
       runSource.close();
       runSource = null;
     }
     currentRunId = null;
   }

   function feedLine(text, className) {
     var div = document.createElement("div");
     div.className = className || "feed-line";
     div.textContent = text;
     document.getElementById("runFeed").appendChild(div);
     return div;
   }

   function openRun(runId) {
     if (!runId) return;
     closeRun();
     currentRunId = runId;
     document.getElementById("runPicker").value = runId;
     var feed = document.getElementById("runFeed");
     feed.innerHTML = "";
     var partial = null;
     var currentCard = null;

     runSource = new EventSource("/api/runs/" + runId + "/stream");
     runSource.onmessage = function(message) {
       if (message.data === "[DONE]") {
         closeRun();
         loadTasks();
         return;
       }
       var event;
       try {
         event = JSON.parse(message.data);
       } catch (err) {
         return;
       }

       if (event.type === "start") {
         feedLine("started by " + event.trigger + " on " + event.model);
       } else if (event.type === "context") {
         feedLine("trimmed " + event.dropped + " old messages to fit the context");
       } else if (event.type === "step") {
         partial = null;
         feedLine("step " + event.step + " of " + event.max_steps);
       } else if (event.type === "token") {
         if (!partial) partial = feedLine("", "feed-partial");
         partial.textContent += event.token;
       } else if (event.type === "tool_call") {
         partial = null;
         currentCard = addToolCardTo(feed, event.name, event.args);
       } else if (event.type === "tool_result") {
         if (currentCard) {
           currentCard.body.textContent = event.result;
           if (event.error) currentCard.card.classList.add("failed");
           currentCard = null;
         }
       } else if (event.type === "final") {
         partial = null;
         feedLine(event.answer || "(empty answer)", "feed-answer");
         if (event.tools_used && event.tools_used.length) {
           feedLine("tools: " + event.tools_used.join(", ") +
                    " | steps: " + event.steps +
                    " | " + Math.round((event.elapsed_ms || 0) / 100) / 10 + "s");
         }
       } else if (event.type === "cancelled") {
         feedLine("cancelled");
       } else if (event.type === "error") {
         feedLine("error: " + event.error, "feed-answer");
       } else if (event.type === "done") {
         feedLine("run finished: " + event.status);
         closeRun();
         loadTasks();
       }
       feed.scrollIntoView({ block: "end" });
     };
     runSource.onerror = function() {
       // The server closes the stream when the run ends. Do not let EventSource
       // reconnect and replay the whole run in a loop.
       closeRun();
     };
   }

   function addToolCardTo(host, name, args) {
     var card = document.createElement("div");
     card.className = "tool-card";
     var head = document.createElement("div");
     head.className = "name";
     head.textContent = "tool: " + name;
     var argsEl = document.createElement("div");
     argsEl.className = "args";
     argsEl.textContent = JSON.stringify(args);
     var body = document.createElement("pre");
     body.textContent = "running...";
     card.appendChild(head);
     card.appendChild(argsEl);
     card.appendChild(body);
     host.appendChild(card);
     return { card: card, body: body };
   }
"""

__all__ = ["UI_JS_TASKS"]
