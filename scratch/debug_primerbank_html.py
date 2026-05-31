import urllib.request
import urllib.parse

def debug_primerbank():
    url = 'https://pga.mgh.harvard.edu/cgi-bin/primerbank/new_search2.cgi'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    post_data = urllib.parse.urlencode({
        'selectBox': 'Keyword',
        'species': 'Human',
        'searchBox': 'kinase',
        'Submit': 'Submit'
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(url, data=post_data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')
        with open("scratch/primerbank_debug.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Wrote scratch/primerbank_debug.html")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    debug_primerbank()
