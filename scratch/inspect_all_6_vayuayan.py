import urllib.request
import gzip
import io
import pandas as pd
import json

stations = {
    'Anand_Vihar': {
        'source_id': 'site_301',
        'file': 'delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz'
    },
    'Bawana': {
        'source_id': 'site_1560',
        'file': 'delhi__delhi__bawana_delhi_dpcc__site_1560.csv.gz'
    },
    'ITO': {
        'source_id': 'site_117',
        'file': 'delhi__delhi__ito_delhi_cpcb__site_117.csv.gz'
    },
    'Mandir_Marg': {
        'source_id': 'site_122',
        'file': 'delhi__delhi__mandir_marg_delhi_dpcc__site_122.csv.gz'
    },
    'Punjabi_Bagh': {
        'source_id': 'site_125',
        'file': 'delhi__delhi__punjabi_bagh_delhi_dpcc__site_125.csv.gz'
    },
    'RK_Puram': {
        'source_id': 'site_124',
        'file': 'delhi__delhi__r_k_puram_delhi_dpcc__site_124.csv.gz'
    }
}

base_url = "https://data.vayuayan.saketlab.org/data/"

results = {}
for name, meta in stations.items():
    url = base_url + meta['file']
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            data = res.read()
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            df = pd.read_csv(gz)
        results[name] = {
            'station_id': meta['source_id'],
            'url': url,
            'file_size': len(data),
            'min_timestamp': df['timestamp'].min(),
            'max_timestamp': df['timestamp'].max(),
            'total_rows': len(df),
            'columns': df.columns.tolist()
        }
    except Exception as e:
        results[name] = {'error': str(e)}

for k, v in results.items():
    print(f"\nStation: {k}")
    print(f"  Station ID: {v.get('station_id')}")
    print(f"  Latest Timestamp: {v.get('max_timestamp')}")
    print(f"  Min Timestamp: {v.get('min_timestamp')}")
    print(f"  Rows: {v.get('total_rows')}")
    print(f"  File Size: {v.get('file_size')} bytes")
