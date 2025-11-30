# youtube_service.py
import os
import subprocess
import pickle
import numpy as np
from tqdm import tqdm
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer
import faiss

class YouTubeService:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_folder="video_indices"):
        self.model = SentenceTransformer(model_name)
        self.index_folder = index_folder
        os.makedirs(self.index_folder, exist_ok=True)

    # -------------------------
    # Search YouTube
    # -------------------------
    def search_video(self, query, max_results=6):
        """Return list of videos (dicts) from youtube search."""
        q = f"{query} explanation for class 7"
        vs = VideosSearch(q, limit=max_results)
        results = vs.result().get("result", [])
        videos = []
        for r in results:
            videos.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "duration": r.get("duration"),
                "channel": r.get("channel", {}).get("name"),
                "views": (r.get("viewCount") or {}).get("short"),
                "link": r.get("link"),
            })
        return videos

    def pick_best_video(self, videos):
        """Basic heuristic to pick the most kid-friendly/short video."""
        if not videos:
            return None
        def score(v):
            s = 0
            title = (v.get("title") or "").lower()
            if any(x in title for x in ["for kids", "class 7", "grade 7", "class7", "cbse", "junior"]):
                s += 10
            # prefer under ~20 minutes
            try:
                parts = list(map(int, (v.get("duration") or "0:00").split(":")))
                mins = parts[-2] + (parts[0] * 60 if len(parts) == 3 else 0)
                if mins <= 20:
                    s += 3
            except Exception:
                pass
            if v.get("views"):
                s += 1
            return s
        return sorted(videos, key=score, reverse=True)[0]

    # -------------------------
    # Transcript fetching
    # -------------------------
    def fetch_transcript_segments(self, video_id, languages=["en"]):
        """
        Return list of segments: {'text','start','duration'}.
        If YouTube transcript unavailable, returns None (caller can fallback to STT).
        """
        try:
            segs = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            # segs already in desired shape
            return segs
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as e:
            print("Transcript fetch error:", e)
            return None

    def download_audio(self, video_url, out_path):
        """
        Download audio using yt-dlp and save to out_path (wav or m4a).
        Requires ffmpeg installed in PATH.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # yt-dlp template: use out_path literal; include extension like .wav or .m4a
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "wav",
            "-o", out_path,
            video_url
        ]
        subprocess.run(cmd, check=True)
        return out_path

    def run_whisper_local(self, audio_path, model="base"):
        """
        Local whisper fallback. Requires 'whisper' package installed.
        Returns list of {'text','start','duration'}.
        """
        import whisper
        w = whisper.load_model(model)
        res = w.transcribe(audio_path)
        segments = []
        for seg in res.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start": float(seg["start"]),
                "duration": float(seg["end"] - seg["start"])
            })
        return segments

    def fetch_transcript_with_fallback(self, video_id, video_url=None, whisper_model="base"):
        """
        Try YouTube transcripts first; if not available, fallback to download + whisper.
        """
        segs = self.fetch_transcript_segments(video_id)
        if segs:
            return segs
        if video_url is None:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        audio_path = os.path.join("downloads", f"{video_id}.wav")
        print("No YouTube transcript — downloading audio and running local STT (this will take time)...")
        self.download_audio(video_url, out_path=audio_path)
        try:
            segments = self.run_whisper_local(audio_path, model=whisper_model)
            return segments
        finally:
            # optionally keep or delete audio; we'll keep for debugging
            pass

    # -------------------------
    # Chunking with timestamps
    # -------------------------
    def chunk_transcript_segments(self, segments, chunk_size_words=150):
        """
        Aggregate segment words until chunk_size_words reached; preserve approximate start and end times.
        Returns list of dicts: {'chunk_id','text','start','end'}
        """
        chunks = []
        words_acc = []
        current_start = None
        current_end = None
        chunk_i = 0

        for seg in segments:
            text = seg.get("text", "").replace("\n", " ").strip()
            if not text:
                continue
            seg_words = text.split()
            if current_start is None:
                current_start = float(seg.get("start", 0.0))
            # append words and update end time
            words_acc.extend(seg_words)
            current_end = float(seg.get("start", 0.0)) + float(seg.get("duration", 0.0))

            if len(words_acc) >= chunk_size_words:
                chunk_i += 1
                chunk_text = " ".join(words_acc)
                chunks.append({
                    "chunk_id": f"video_{chunk_i}",
                    "text": chunk_text,
                    "start": current_start,
                    "end": current_end
                })
                words_acc = []
                current_start = None
                current_end = None

        # leftover
        if words_acc:
            chunk_i += 1
            chunks.append({
                "chunk_id": f"video_{chunk_i}",
                "text": " ".join(words_acc),
                "start": current_start or 0.0,
                "end": current_end or 0.0
            })
        return chunks

    # -------------------------
    # Build per-video FAISS index and metadata
    # -------------------------
    def build_and_save_video_index(self, video_id, segments, chunk_size_words=150, model_name=None):
        """
        segments: list of {'text','start','duration'}
        Saves:
          video_indices/video_{video_id}.index
          video_indices/video_{video_id}_metadata.pkl   (mapping idx -> {'chunk_id','text','start','end'})
        Returns paths
        """
        # chunk
        chunks = self.chunk_transcript_segments(segments, chunk_size_words=chunk_size_words)
        texts = [c["text"] for c in chunks]
        if model_name is None:
            model = self.model
        else:
            model = SentenceTransformer(model_name)

        # embeddings
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype="float32")

        # build FAISS
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # save
        idx_path = os.path.join(self.index_folder, f"video_{video_id}.index")
        meta_path = os.path.join(self.index_folder, f"video_{video_id}_metadata.pkl")
        faiss.write_index(index, idx_path)

        # metadata: store as dict keyed by index position (0..n-1)
        metadata = {}
        for i, c in enumerate(chunks):
            metadata[i] = {"chunk_id": c["chunk_id"], "text": c["text"], "start": c["start"], "end": c["end"]}
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        return idx_path, meta_path
