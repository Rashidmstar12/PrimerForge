import urllib.request
import urllib.error

names = [
    "primer-forge",
    "primerforge-pangenome",
    "primerforge-py",
    "primerforge-ml",
    "primerforge-bio",
    "primerforge-suite",
    "primerforge-biophysics"
]

for n in names:
    try:
        urllib.request.urlopen(f"https://pypi.org/pypi/{n}/json")
        print(f"{n}: TAKEN")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"{n}: AVAILABLE")
        else:
            print(f"{n}: ERROR {e.code}")
