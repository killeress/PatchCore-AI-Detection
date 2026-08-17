import os
import subprocess
import time
from PIL import Image

def main():
    html_path = os.path.abspath(r"D:\SourceCode\PatchCore-AI-Detection\reports\slide_20260811_0814.html")
    output_png = os.path.abspath(r"D:\SourceCode\PatchCore-AI-Detection\reports\slide_20260811_0814_raw.png")
    output_jpg = os.path.abspath(r"D:\SourceCode\PatchCore-AI-Detection\reports\slide_20260811_0814.jpg")

    # Find browser
    browser_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    browser = next((p for p in browser_candidates if os.path.exists(p)), None)
    if not browser:
        raise RuntimeError("No Chrome or Edge browser found!")

    url = f"file:///{html_path.replace(os.sep, '/')}"
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1920,920",
        "--force-device-scale-factor=1",
        f"--screenshot={output_png}",
        url
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(output_png):
        raise RuntimeError(f"Failed to generate {output_png}")

    img = Image.open(output_png)
    print(f"Captured PNG size: {img.size}, mode: {img.mode}")

    # Convert to RGB and save as high quality JPG
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img.save(output_jpg, "JPEG", quality=96, subsampling=0)
    print(f"Successfully saved JPG to: {output_jpg}")

    # Copy to artifact folder for easy preview in markdown if needed
    artifact_jpg = r"C:\Users\rh.syu\.gemini\antigravity-cli\brain\65de1adc-e8fc-45d9-9a3e-ceaaf0ba77c8\slide_20260811_0814.jpg"
    img.save(artifact_jpg, "JPEG", quality=96, subsampling=0)
    print(f"Copied to artifact directory: {artifact_jpg}")

if __name__ == "__main__":
    main()
