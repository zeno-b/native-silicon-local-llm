"""The embedded single-page UI and its render/validation helpers.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from .core import *  # noqa: F401,F403
from .ui_styles import UI_CSS
from .ui_markup import UI_BODY, UI_HEAD, UI_TAIL
from .ui_script_chat import UI_JS_CHAT
from .ui_script_models import UI_JS_MODELS
from .ui_script_panels import UI_JS_PANELS
from .ui_script_tasks import UI_JS_TASKS
from .ui_script_views import UI_JS_VIEWS
from .ui_script_auth import UI_JS_AUTH


# Texcel Solutions brand mark (the hexagonal molecule), used as the default logo
# when APP_LOGO is not set. Cleaned from the site's SVG: the redundant <style>
# blocks are dropped (the page must keep exactly one <style>), the gradient id is
# namespaced, and the fill is set inline on the path so the warm brand gradient
# renders without them. Sized by the .brand-logo rule.
TEXCEL_LOGO_SVG = (
    '<svg class="brand-logo" viewBox="0 0 160.03026 183.20361"'
    ' width="160" height="183"'
    ' xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Texcel logo">'
    '<defs><linearGradient id="__GID__" x1="57.349998" y1="95.660004"'
    ' x2="224.47" y2="95.660004" gradientUnits="userSpaceOnUse">'
    '<stop offset="0" stop-color="#fee10a"/>'
    '<stop offset="0.2" stop-color="#f7b015"/>'
    '<stop offset="0.5" stop-color="#f3921a"/>'
    '<stop offset="0.8" stop-color="#e85a18"/>'
    '<stop offset="1" stop-color="#d83027"/>'
    '</linearGradient></defs>'
    '<g transform="matrix(0.95757694,0,0,0.95757694,-54.917038,0)"'
    ' style="fill:url(#__GID__)"><path style="fill:url(#__GID__)" d="'
    'm 224.47,112.17 v -6.79 l -5.88,-3.38 -5.89,3.39 v 2.73 l -19,1.66 v -3.35 l -5.9,-3.43 -5.88,3.39 v 1.73 l -27.37,-9.77 -1.66,-6.81 29.88,-18.36 v 1.85 l 9.74,5.62 -3.33,23.09 5.61,-23.87 7,-4 9,4.2 -7.61,-7 V 63.26 L 201,62 212.52,54.92 v 2.25 l 3.53,2 2.35,42.65 2.35,-42.65 3.53,-2 v -6.78 l -5.87,-3.39 -4.53,2.62 -7.75,-4.79 V 39 l -5.88,-3.39 -4.7,2.71 -48.9,-30.2 v -4.7 0 0 L 140.8,0 135,3.36 v 6.71 0 0 l 1,0.55 -18.76,10.54 v -1.27 l -5.88,-3.4 -5.88,3.4 v 6.79 l 0.93,0.53 -39.3,22 -3.88,-2.24 -5.88,3.39 v 6.79 L 61,59.22 62.92,99 57.38,102.19 V 109 l 5.54,3.19 -0.49,22.05 -5,2.9 v 6.79 l 5.88,3.4 5.88,-3.4 v -2.65 l 14.93,-0.66 -4.87,10.77 -10.44,-5.39 64,35.63 v 5.09 l 8,4.6 0.12,-0.07 8,-4.6 v -3.76 l 63,-36.39 4,2.31 5.88,-3.4 v -6.79 l -4.25,-2.45 2.06,-21.1 z m -66.29,11.46 -15.44,-6.78 0.43,-12.63 8,-4.6 v -7 l 0.39,-0.24 z m 24.6,-53.41 -31.16,16.06 -3.3,-13.54 10.68,-13.63 2.53,1.46 5.88,-3.39 V 55.66 L 182.78,64 Z m -39.56,-18.44 -2.57,-10.55 15.94,8.6 -1,0.56 v 6.79 l 2.27,1.31 -9.7,13.61 -4.64,-19.1 11.67,-0.6 z m -12.54,-6.44 -2.61,1.5 -10.19,-10 10.12,-2.45 12.65,6.83 2.25,10.55 -6.53,-0.35 v -2.81 z m 71.6,2.66 3.84,-2.21 7.45,4 -1,0.61 v 4.52 L 200,61.39 199.28,60.97 Z m -7.91,-8.55 v 6.33 l 2.95,1.71 -1.65,11.45 -2.67,-1.57 -10.21,5.88 v 0.44 L 167.4,54.81 v -4.42 l -1.08,-0.64 14,-17.91 z M 177.88,30.56 164.77,48.88 161.51,47 157.66,49.22 130.78,33.7 167.15,24.76 Z M 117.24,22 136,10.72 l 4.8,2.77 2.6,-1.5 20.3,10.94 -34.04,10.12 -12.42,-7.16 z m -52,37.4 3.91,-2.25 v -6.39 l 38.34,-22.94 3.91,2.27 4.15,-2.42 11.45,6.18 -9.34,2.77 -3.33,-3.26 2.85,3.4 -5.46,1.63 5.62,-1.39 9.06,10.81 -1.38,0.81 v 6.55 l 2.72,1.57 -15,10.87 -6.27,-3.61 -0.46,0.26 -8,-28.83 6.19,29.87 -6.12,3.51 v 1.51 L 70,54.82 98.12,71.6 v 6.88 l 0.69,0.41 -16.13,11.6 -18.59,-8.17 z m 72.76,44.4 0.38,11.13 -16.68,-7.33 9.06,-10.44 v 2.46 z m -16.65,26.29 -1.33,0.77 -3.85,-16.86 4.91,-5.65 17.29,7.2 0.76,22.4 -9.72,0.19 v -3.35 z M 102.79,99.31 83.64,90.92 98.93,79 l 7.54,4.34 2.16,-1.25 0.49,3.26 z m 6.34,-14 2.71,17.94 -8.65,-3.81 z m 2.9,19.18 1.73,11.37 -12.59,15.36 12.7,-14.64 2.47,16.42 -3.06,1.76 v 3.68 l -26.6,0.53 16,-38.29 z m 2.21,0.92 6,2.51 -4.26,5.2 v 0 z m 16.5,-10.29 -9.9,12.07 -6.89,-3 -5.06,-22.29 3.3,-1.91 18.55,11.07 z m -67.41,3.3 0.67,-13.89 17.07,7.13 -13.45,9.66 -4.29,-2.48 z m 5.83,40.89 v -2.17 l -5,-2.9 -0.49,-22.05 5.49,-3.19 v -6.76 l 13,-10.15 20.12,8.38 L 84.83,139 Z m 11.9,13 5,-11.81 27.26,-1.2 v 4.73 l 5.19,3 4.06,26.9 z m 51.75,25.14 v 1.83 l -2,-1 -7,-30.93 5.66,-3.27 v -5.46 l 9.73,-0.43 1.18,35 z m -2.06,-89.66 v 0.54 L 113.86,79 l 1,-0.55 v -9.1 l 13,-12.56 2.84,1.64 5.68,-3.28 v -1.78 l 6.79,-0.34 4.28,20.1 -6.35,8.87 6.46,-8.26 2.81,13.2 -0.42,0.22 L 141,82 Z m 18.1,90.21 v -0.54 l -8,-4.6 v 0 l 1.15,-34.81 9.11,-0.38 -9.1,0.18 0.69,-20.53 15.48,6.45 9.52,44.74 z m 22.34,-11.21 -10.19,-41.86 2.75,1.15 -2.75,-1.21 -6.4,-26.34 27.32,11.76 v 2.93 l 4.26,2.46 -6.85,40.84 8.3,-40 0.18,0.1 3.76,-2.18 20.17,23.12 -1.79,1 v 6.79 l 1.82,1.06 z m 46,-30.75 -1.42,-0.82 -2.36,1.36 -21.72,-22.22 2,-1.13 v -3.39 l 19,-0.16 v 2.51 l 4.83,2.79 z'
    '"/></g></svg>'
)


def render_ui() -> str:
    """Return the HTML the browser receives, with the build marker substituted.

    HTML_PAGE is a raw string on purpose. In a normal Python string, a `\\n`
    written inside embedded JavaScript becomes a real line break, which splits a
    JS string literal across two lines and makes the whole <script> fail to
    parse. The page then renders but nothing works: the status panel sits on its
    hardcoded "Starting..." text and the Send button does nothing.
    """
    # Render the logo as an <img> when it looks like a URL/path, otherwise inline
    # it as text or an emoji. Escape the name so a stray < in APP_NAME cannot
    # break the header markup.
    import html as _html
    name = _html.escape(APP_NAME)
    logo = APP_LOGO.strip()
    if logo.startswith(("http://", "https://", "/")):
        logo_html = f'<img src="{_html.escape(logo)}" alt="{name} logo" class="brand-logo">'
        logo_login = logo_html
    elif logo:
        logo_html = f'<span class="brand-mark">{_html.escape(logo)}</span>'
        logo_login = logo_html
    else:
        # The default brand SVG appears twice (header + login box). Each needs its
        # OWN gradient id: SVG paint servers are resolved by id, and the header's
        # is inside a display:none subtree while the login screen is up, so a
        # shared id would leave the login logo unpainted. Distinct ids fix it.
        logo_html = TEXCEL_LOGO_SVG.replace("__GID__", "txLogoGrad")
        logo_login = TEXCEL_LOGO_SVG.replace("__GID__", "txLogoGradLogin")
    return (HTML_PAGE
            .replace("{{UI_BUILD}}", UI_BUILD)
            .replace("{{APP_NAME}}", name)
            .replace("{{APP_LOGO_LOGIN}}", logo_login)
            .replace("{{APP_LOGO}}", logo_html))


def check_ui_syntax() -> list[str]:
    """Cheap structural check on the rendered <script>, no Node required.

    Catches the failure above by scanning the script as a character stream and
    reporting a real newline inside a single- or double-quoted literal. The
    previous version counted quotes per line, which flagged any correct line
    containing an apostrophe ("it's") and then refused to start the server. A
    scanner that tracks comments, escapes and template literals has no such
    false positive. Returns a list of problems; empty means well formed.
    """
    script = re.search(r"<script>(.*?)</script>", render_ui(), re.S)
    if not script:
        return ["no <script> block found in HTML_PAGE"]
    return scan_js_strings(script.group(1))


def scan_js_strings(body: str) -> list[str]:
    """Report string literals broken by a real newline, and unclosed comments."""
    problems: list[str] = []
    quote: str | None = None
    quote_line = 0
    line = 1
    index = 0
    length = len(body)

    while index < length:
        char = body[index]
        nxt = body[index + 1] if index + 1 < length else ""

        if char == "\n":
            line += 1
            if quote in ('"', "'"):
                problems.append(f"line {quote_line}: unterminated {quote} string literal")
                quote = None
            index += 1
            continue

        if quote is None:
            if char == "/" and nxt == "/":
                while index < length and body[index] != "\n":
                    index += 1
                continue
            if char == "/" and nxt == "*":
                end = body.find("*/", index + 2)
                if end == -1:
                    problems.append(f"line {line}: unterminated block comment")
                    break
                line += body.count("\n", index, end)
                index = end + 2
                continue
            if char in ('"', "'", "`"):
                quote = char
                quote_line = line
                index += 1
                continue
            if char == "/":
                # A regex literal. Distinguish it from division by looking at the
                # previous significant character: after a value (identifier, digit,
                # closing bracket) a slash is division; otherwise it starts a regex.
                prev = ""
                back = index - 1
                while back >= 0 and body[back] in " \t\n":
                    back -= 1
                if back >= 0:
                    prev = body[back]
                if prev and (prev.isalnum() or prev in ")]_$"):
                    index += 1
                    continue
                scan = index + 1
                in_class = False
                while scan < len(body):
                    ch = body[scan]
                    if ch == "\\":
                        scan += 2
                        continue
                    if ch == "\n":
                        break
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        break
                    scan += 1
                if scan < len(body) and body[scan] == "/":
                    line += body.count("\n", index, scan)
                    index = scan + 1
                    continue
            index += 1
            continue

        if char == "\\":
            index += 2
            continue
        if char == quote:
            quote = None
        index += 1

    if quote is not None:
        problems.append(f"line {quote_line}: unterminated {quote} string literal at end of script")
    return problems

HTML_PAGE = (
    UI_HEAD
    + UI_CSS
    + UI_BODY
    # One <script> block, assembled from parts. JavaScript hoists function
    # declarations across the whole script, so splitting the source here changes
    # nothing about how it runs — the browser still receives a single script.
    + UI_JS_CHAT
    + "\n" + UI_JS_VIEWS
    + "\n" + UI_JS_TASKS
    + "\n" + UI_JS_MODELS
    + "\n" + UI_JS_AUTH
    + "\n" + UI_JS_PANELS
    + UI_TAIL
)



# Request/response models are bound at module scope by _define_api_models().
# They cannot be plain module-level class statements because pydantic is not
# installed until bootstrap() has run.
ChatRequest: Any = None
FeedbackRequest: Any = None
ChatResponse: Any = None
ConfigRequest: Any = None
ToolRequest: Any = None
MemoryRequest: Any = None
TaskRequest: Any = None
TaskUpdateRequest: Any = None
ModelSelectRequest: Any = None



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ChatRequest',
    'ChatResponse',
    'ConfigRequest',
    'FeedbackRequest',
    'HTML_PAGE',
    'MemoryRequest',
    'ModelSelectRequest',
    'TaskRequest',
    'TaskUpdateRequest',
    'ToolRequest',
    'check_ui_syntax',
    'render_ui',
    'scan_js_strings',
]
