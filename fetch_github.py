import requests
import sys

urls = [
    ("cansik/realsense-pose-detector", "https://raw.githubusercontent.com/cansik/realsense-pose-detector/main/README.md"),
    ("cansik/realsense-pose-detector", "https://raw.githubusercontent.com/cansik/realsense-pose-detector/master/README.md"),
    ("SiaMahmoudi/MediaPipe-pose-estimation", "https://raw.githubusercontent.com/SiaMahmoudi/MediaPipe-pose-estimation-using-intel-realsense-debth-camera/main/README.md"),
    ("SiaMahmoudi/MediaPipe-pose-estimation", "https://raw.githubusercontent.com/SiaMahmoudi/MediaPipe-pose-estimation-using-intel-realsense-debth-camera/master/README.md"),
    ("Razg93/Skeleton-Tracking", "https://raw.githubusercontent.com/Razg93/Skeleton-Tracking-using-RealSense-depth-camera/main/README.md"),
    ("Razg93/Skeleton-Tracking", "https://raw.githubusercontent.com/Razg93/Skeleton-Tracking-using-RealSense-depth-camera/master/README.md"),
]

for name, url in urls:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            print(f"=== {name} from {url} ===")
            print(r.text[:5000])
            print("\n\n")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
