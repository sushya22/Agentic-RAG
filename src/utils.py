def safe_truncate(s, n=400):
    return s if len(s)<=n else s[:n].rsplit(' ',1)[0] + '...'
