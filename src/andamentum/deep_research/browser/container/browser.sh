#!/bin/sh
# Headless Chromium binds its debugging port to 127.0.0.1 only (it ignores
# --remote-debugging-address in headless mode); socat forwards the published
# port 9222 to it.
socat TCP-LISTEN:9222,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:9223 &
# Headless Chromium announces itself as "HeadlessChrome" and DuckDuckGo then
# answers every search with a bot check. The user agent is the installed
# version's ordinary desktop one.
major=$(chromium --version | grep -oE '[0-9]+' | head -1)
agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${major}.0.0.0 Safari/537.36"
exec chromium --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage --user-data-dir=/profile --remote-debugging-port=9223 --no-first-run --no-default-browser-check --disable-extensions --user-agent="$agent" about:blank
