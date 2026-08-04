#!/usr/bin/env python3
"""Patient watcher: poll ComfyUI history for a specific prompt_id until it
completes (success or error) or an optional hard cap is reached. Prints a
single definitive line + the produced file. No misleading early timeout."""
import json, sys, time, urllib.request

HOST = "http://127.0.0.1:8188"
PID = sys.argv[1] if len(sys.argv) > 1 else "e19f1156"
CAP_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 40  # default 40 min cap

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())

start = time.time()
while time.time() - start < CAP_MIN * 60:
    time.sleep(15)
    try:
        h = get(HOST + "/history/" + PID)
    except Exception:
        continue
    if PID in h:
        st = h[PID].get("status", {})
        if st.get("status_str") == "success" or st.get("completed"):
            outs = h[PID].get("outputs", {})
            files = []
            for nid, o in outs.items():
                for k, v in o.items():
                    if isinstance(v, list):
                        files += [x.get("filename") for x in v if isinstance(x, dict) and "filename" in x]
                    elif isinstance(v, dict) and "filename" in v:
                        files.append(v["filename"])
            print("SUCCESS_PID=" + PID)
            print("FILES=" + json.dumps(files))
            sys.exit(0)
        errs = [m for _, m in st.get("messages", []) if m.get("type") == "execution_error"]
        if errs:
            print("ERROR_PID=" + PID)
            print(json.dumps(errs[0].get("data", {}))[:2500])
            sys.exit(1)
        # 'executing'/'execution_interrupted'/end states may be present but no completion yet
        print("still-running elapsed_min=%.1f" % ((time.time() - start) / 60))
    else:
        pass  # not written to history until done

print("CAP_REACHED_MINS=%d pid=%s" % (CAP_MIN, PID))
sys.exit(1)
