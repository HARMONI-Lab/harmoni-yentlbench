try:
    from scipy.stats import false_discovery_control
    print("scipy.stats.false_discovery_control available")
except ImportError:
    print("Not available in scipy")
