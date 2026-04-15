"""
Script to automate multiple HTTP POST requests for BTS
Aviation On-Time Performance Database downloads.

Downloads data from January 2022 to December 2025.
URL: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr
"""

import requests
import zipfile
import io
import sys
import os
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Extract zip files to data/ontime/ at repo root
SCRIPT_DIR = Path(__file__).parent.parent
zpath = SCRIPT_DIR / 'data' / 'ontime'

# Base download page URL
BASE_URL = 'https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr'

# Request headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def get_viewstate_tokens(session, retries=3):
    """
    Fetch the page and extract ViewState and EventValidation tokens.
    
    Params:
        session: requests.Session object
        retries: number of retry attempts
    
    Returns:
        Tuple of (viewstate, viewstategenerator, eventvalidation) or (None, None, None)
    """
    for attempt in range(retries):
        try:
            print(f'  Fetching form tokens (attempt {attempt + 1}/{retries})...', end=' ')
            r = session.get(BASE_URL, headers=HEADERS, timeout=30)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Extract ASP.NET form fields
            viewstate = soup.find('input', {'name': '__VIEWSTATE'})
            viewstategenerator = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})
            eventvalidation = soup.find('input', {'name': '__EVENTVALIDATION'})
            
            if viewstate and viewstategenerator and eventvalidation:
                print('✓')
                return (viewstate['value'], viewstategenerator['value'], eventvalidation['value'])
            else:
                print('✗ Missing form fields')
                if attempt < retries - 1:
                    time.sleep(5)
        except Exception as e:
            print(f'✗ {type(e).__name__}')
            if attempt < retries - 1:
                time.sleep(5)
    
    return (None, None, None)

def httprequest(session, month_num, year, retries=3):
    """
    Function to send POST request with ViewState tokens.
    
    Params:
        session: requests.Session object
        month_num: month number (1-12)
        year: year as string
        retries: number of retry attempts
    
    Returns:
        Response content (bytes) or None
    """
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    month_name = months[month_num - 1]
    
    for attempt in range(retries):
        try:
            # Get fresh tokens for each request
            viewstate, viewstategenerator, eventvalidation = get_viewstate_tokens(session)
            
            if not all([viewstate, viewstategenerator, eventvalidation]):
                print(f'  ✗ Could not retrieve form tokens')
                continue
            
            # Build POST data
            post_data = {
                '__EVENTTARGET': '',
                '__EVENTARGUMENT': '',
                '__LASTFOCUS': '',
                '__VIEWSTATE': viewstate,
                '__VIEWSTATEGENERATOR': viewstategenerator,
                '__EVENTVALIDATION': eventvalidation,
                'txtSearch': '',
                'cboGeography': 'All',
                'cboYear': year,
                'cboPeriod': str(month_num),
                'btnDownload': 'Download',
                'YEAR': 'on',
                'DAY_OF_WEEK': 'on',
                'FL_DATE': 'on',
                'OP_UNIQUE_CARRIER': 'on',
                'ORIGIN_AIRPORT_ID': 'on',
                'ORIGIN': 'on',
                'DEST_AIRPORT_ID': 'on',
                'DEST': 'on',
                'DEP_TIME': 'on',
                'DEP_DELAY': 'on',
                'ARR_TIME': 'on',
                'ARR_DELAY': 'on',
                'CANCELLED': 'on',
                'CANCELLATION_CODE': 'on',
                'DIVERTED': 'on',
                'CARRIER_DELAY': 'on',
                'WEATHER_DELAY': 'on',
                'NAS_DELAY': 'on',
                'SECURITY_DELAY': 'on',
                'LATE_AIRCRAFT_DELAY': 'on'
            }
            
            print(f'  Downloading {month_name} {year}...', end=' ')
            r = session.post(BASE_URL, headers=HEADERS, data=post_data, timeout=60, allow_redirects=True)
            
            # Check if response is a zip file (starts with PK magic bytes)
            if r.status_code == 200 and r.content[:2] == b'PK':
                print('✓')
                return r.content
            elif r.status_code == 200:
                print(f'✗ Response not a zip file (size: {len(r.content)} bytes)')
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f'    Retrying in {wait_time}s...')
                    time.sleep(wait_time)
            else:
                print(f'✗ HTTP {r.status_code}')
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f'    Retrying in {wait_time}s...')
                    time.sleep(wait_time)
                    
        except requests.exceptions.RequestException as e:
            print(f'✗ {type(e).__name__}')
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f'    Retrying in {wait_time}s...')
                time.sleep(wait_time)
    
    print(f'✗ Failed to download {month_name} {year} after {retries} attempts')
    return None

def extractZipToDf(content):
    """
    Extract a zip response and return its contents as a DataFrame.

    Params:
        content: response content (bytes)

    Returns:
        pd.DataFrame or None
    """
    import pandas as pd
    if content is None:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(content))
        csvFiles = [f for f in z.namelist() if f.endswith('.csv')]
        if not csvFiles:
            print('  ✗ No CSV found in zip')
            return None
        dfs = [pd.read_csv(z.open(f), low_memory=False) for f in csvFiles]
        return pd.concat(dfs, ignore_index=True) if dfs else None
    except Exception as e:
        print(f'  ✗ Error reading zip: {e}')
        return None

def main():
    """Download BTS on-time data for 2022-2025, saving one CSV per year."""
    import pandas as pd

    years = ['2022', '2023', '2024', '2025']
    monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']

    print("=" * 70)
    print("BTS On-Time Performance Data Download (2022-2025)")
    print("One output CSV per year")
    print("=" * 70)
    print(f"Output folder: {zpath}\n")

    Path(zpath).mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    totalMonths = 0
    failedMonths = []

    for y in years:
        print(f"\nProcessing year {y}...")
        yearParts = []

        for m in range(1, 13):
            content = httprequest(session, m, y)
            if content:
                df = extractZipToDf(content)
                if df is not None:
                    yearParts.append(df)
                    totalMonths += 1
            else:
                failedMonths.append(f"{monthNames[m - 1]} {y}")

        # Concatenate all months and save as a single CSV for the year
        if yearParts:
            yearDf = pd.concat(yearParts, ignore_index=True)
            outPath = zpath / f'ontime_{y}.csv'
            yearDf.to_csv(outPath, index=False)
            print(f"  Saved {len(yearDf):,} rows -> {outPath}")
        else:
            print(f"  No data downloaded for {y}")

    print("\n" + "=" * 70)
    print(f"Download complete!")
    print(f"  Successfully downloaded: {totalMonths} months")
    if failedMonths:
        print(f"  Failed: {len(failedMonths)} months")
        for month in failedMonths:
            print(f"    - {month}")
    print("=" * 70)

# Call main function
if __name__ == '__main__':
    main()
