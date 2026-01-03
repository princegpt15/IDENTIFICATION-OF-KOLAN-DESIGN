# BULK IMAGE DOWNLOAD GUIDE

## Method 1: Chrome/Edge Extension (RECOMMENDED - Easiest)

### Step 1: Install Extension
1. Open Chrome or Edge browser
2. Go to: https://chrome.google.com/webstore
3. Search for: **"Download All Images"** or **"Image Downloader"**
4. Click **"Add to Chrome/Edge"**
5. Click **"Add extension"**

### Step 2: Use the Extension
1. Go to Google Images search (tabs already open)
2. **Scroll down** to load 100-150 images
3. **Click the extension icon** in your browser toolbar (top-right)
4. Select **"Download all images"** or filter by minimum size (e.g., 200x200)
5. Choose save location: `C:\Users\princ\Desktop\MACHINE TRAINING\kolam_dataset\00_raw_data\pulli_kolam\`
6. Click **Download** - all images download at once!

**Popular Extensions:**
- **Download All Images** - Simple, one-click download
- **Image Downloader** - Filter by size, format
- **Bulk Image Downloader** - Advanced filtering

---

## Method 2: Firefox Extension

### Step 1: Install DownThemAll
1. Open Firefox
2. Go to: https://addons.mozilla.org
3. Search: **"DownThemAll"**
4. Click **"Add to Firefox"**

### Step 2: Use DownThemAll
1. Right-click on the Google Images page
2. Select **"DownThemAll!"**
3. Filter: Select only images (jpg, png)
4. Choose folder
5. Click **Start** - downloads all at once

---

## Method 3: Built-in Browser Developer Tools (No Extension)

### For Chrome/Edge:
1. Open Google Images search
2. Scroll to load 100+ images
3. Press **F12** (opens Developer Tools)
4. Click **Console** tab
5. Paste this code and press Enter:

```javascript
// Download all visible images
let imgs = document.querySelectorAll('img[src*="gstatic"]');
let urls = Array.from(imgs).map(img => img.src);
urls.forEach((url, i) => {
    fetch(url)
        .then(res => res.blob())
        .then(blob => {
            let a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `kolam_${i+1}.jpg`;
            a.click();
        });
});
```

**Note:** This triggers multiple downloads. Your browser will ask to "Allow multiple downloads" - click **Allow**.

---

## Method 4: Python Script (Automated)

I can create a Python script that downloads images automatically. However, Google has rate limits, so it's slower but more reliable.

Would you like me to create this?

---

## RECOMMENDED WORKFLOW:

### Option A: Browser Extension (Fastest - 5 minutes per category)
1. Install "Download All Images" extension
2. Open the 4 tabs I already opened
3. For each tab:
   - Scroll to load 100 images
   - Click extension icon
   - Select "Download all images"
   - Choose the correct category folder
   - Wait 30-60 seconds
4. **Done! 400 images in ~20 minutes**

### Option B: Manual but Smart (10-15 minutes per category)
1. In Google Images, scroll to load images
2. Hold **Ctrl** and click 10-20 image thumbnails (opens in new tabs)
3. Use extension on each tab or save manually
4. Repeat until you have 50+ per category

---

## AFTER DOWNLOADING:

Run this to check your progress:
```powershell
Get-ChildItem "kolam_dataset\00_raw_data" -Directory | ForEach-Object { 
    $c = (Get-ChildItem $_.FullName -File).Count
    Write-Host "$($_.Name): $c images" 
}
```

**Target:** 50-100 images per category (200-400 total)

Then I'll automatically train the model!
