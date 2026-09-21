"""Read OS-reported lifetime peak memory for this process (no extra dependency)."""
import json
import sys


def peak_process_memory():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "peak_rss_bytes": int(counters.PeakWorkingSetSize),
            "peak_committed_bytes": int(counters.PeakPagefileUsage),
            "memory_measurement": "Windows GetProcessMemoryInfo: lifetime peak working set and commit",
        }
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "peak_committed_bytes": None,
            "memory_measurement": "getrusage(RUSAGE_SELF): lifetime peak RSS",
        }
    raise RuntimeError(f"Peak memory measurement is not implemented for {sys.platform}")


if __name__ == "__main__":
    print(json.dumps(peak_process_memory(), indent=2))
