"""Read Menia's Apple record and increment this version's build; never create records."""
import json
import re
import subprocess
from pathlib import Path

BUNDLE_ID = "com.meniaapp.menia"
VERSION = "0.2.0"


def apple(*args):
    return subprocess.check_output(
        ["app-store-connect", *args, "--no-color"], text=True
    ).strip()


apps = json.loads(apple("apps", "list", "--bundle-id-identifier", BUNDLE_ID,
                        "--strict-match-identifier", "--json"))
if len(apps) != 1:
    raise SystemExit("Créer ou vérifier la fiche Menia (com.meniaapp.menia) dans App Store Connect.")
app_id = apps[0]["id"]
latest = apple("get-latest-testflight-build-number", app_id,
               "--pre-release-version", VERSION, "--platform", "IOS")
# The CLI returns an empty result when no build exists. Never hide an API failure.
if latest and not re.fullmatch(r"[0-9]+", latest):
    raise SystemExit(f"Numéro de build Apple non entier à vérifier : {latest!r}")
next_build = int(latest or "0") + 1
subprocess.run(["xcrun", "agvtool", "new-version", "-all", str(next_build)],
               cwd=Path(__file__).resolve().parent, check=True)
print(f"Menia {VERSION} ({next_build}) — https://appstoreconnect.apple.com/apps/{app_id}/testflight")
