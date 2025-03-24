import yt_dlp
import urllib.parse
import time
import os

# SEARCH_QUERY = "squat exercise side view"
SEARCH_QUERY = "bad squat form side view"
MAX_RESULTS = 10
MAX_DURATION_SECONDS = 60
DOWNLOAD_DIR = "squat_videos"

def fast_search_then_filter(query, max_results=10, max_duration=60):
    query_encoded = urllib.parse.quote(query)
    search_url = f"https://www.youtube.com/results?search_query={query_encoded}"

    print(f"\n🔍 [START] 검색어: {query}")
    print(f"🔗 검색 URL: {search_url}\n")

    # 1단계: 빠른 ID 추출용 검색 (flat)
    flat_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'noplaylist': True,
    }

    candidates = []
    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        try:
            info = ydl.extract_info(search_url, download=False)
            for entry in info.get("entries", []):
                if "shorts" in entry.get("url", ""):
                    continue
                video_id = entry.get("id")
                title = entry.get("title", "No Title")
                url = f"https://www.youtube.com/watch?v={video_id}"
                candidates.append({"title": title, "url": url})
                if len(candidates) >= max_results * 2:  # 여유롭게 추출
                    break
        except Exception as e:
            print(f"❌ [ERROR] 검색 실패: {e}")
            return

    # 2단계: duration 기준 필터링
    print(f"\n⏱️ 영상 길이 확인 중... (최대 {max_results}개)\n")
    filtered = []
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for entry in candidates:
            try:
                info = ydl.extract_info(entry["url"], download=False)
                duration = info.get("duration", 0)
                if duration and duration <= max_duration:
                    print(f"{len(filtered)+1}. {entry['title']} ({duration}s)")
                    filtered.append({
                        "title": entry["title"],
                        "url": entry["url"]
                    })
                else:
                    print(f"⏩ SKIP (길이 {duration}s): {entry['title']}")
            except Exception as e:
                print(f"⚠️ 영상 분석 실패: {entry['url']} - {e}")

            if len(filtered) >= max_results:
                break

    # 3단계: 다운로드
    print(f"\n⬇️ 최종 {len(filtered)}개 영상 다운로드 중...\n")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for idx, video in enumerate(filtered, 1):
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in video["title"])
        output_template = os.path.join(DOWNLOAD_DIR, f"{idx}_{safe_title}.%(ext)s")

        download_opts = {
            'outtmpl': output_template,
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'restrict_filenames': True,
            'force_generic_extractor': False
        }

        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.download([video["url"]])
        except Exception as e:
            print(f"⚠️ 다운로드 실패: {video['url']} - {e}")

if __name__ == "__main__":
    start = time.time()
    fast_search_then_filter(SEARCH_QUERY, MAX_RESULTS, MAX_DURATION_SECONDS)
    print(f"\n✅ 완료! 전체 소요 시간: {round(time.time() - start, 2)}초")