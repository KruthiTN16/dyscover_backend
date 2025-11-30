# rag_youtube.py
"""
Backend YouTube RAG:
- search_youtube(query) → list of videos
- prepare_video(video_id) → build & cache transcripts, chunks, embeddings
- ask_video(question, video_id, timestamp=None) → returns LLM answer

Transcripts NEVER exposed to UI.
"""

import os
from typing import List, Dict, Any, Optional

# --- FFmpeg Path Fix (Windows) ---
DEFAULT_FFMPEG = r"C:\Users\kruth\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + os.getenv("FFMPEG_PATH", DEFAULT_FFMPEG)

class YouTubeRAG:
    def __init__(
        self,
        embed_model_name="all-MiniLM-L6-v2",
        whisper_size="base",
        llm_model="llama3.2:1b",  # ⭐ small model that WILL run on your RAM
        ffmpeg_location=None,
    ):
        self.embed_model_name = embed_model_name
        self.whisper_size = whisper_size
        self.llm_model = llm_model
        self.ffmpeg_location = ffmpeg_location

        self._videos: Dict[str, Dict[str, Any]] = {}

        self._embedder = None
        self._whisper = None

        print("YouTubeRAG backend initialized.")

    # ----------------------------------------------------
    # LAZY LOADERS
    # ----------------------------------------------------
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _get_whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(self.whisper_size)
        return self._whisper

    def _llm_call(self, prompt: str) -> str:
        from ollama import chat
        try:
            resp = chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            return resp.message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    # ----------------------------------------------------
    # 1) SEARCH YOUTUBE
    # ----------------------------------------------------
    def search_youtube(self, query: str, max_results: int = 5):
        from yt_dlp import YoutubeDL

        search_query = f"ytsearch{max_results}:{query}"
        ydl_opts = {"quiet": True, "extract_flat": "in_playlist"}

        if self.ffmpeg_location:
            ydl_opts["ffmpeg_location"] = self.ffmpeg_location

        with YoutubeDL(ydl_opts) as ydl:
            data = ydl.extract_info(search_query, download=False)

        return [
            {
                "title": e.get("title"),
                "video_id": e.get("id"),
                "channel": e.get("channel"),
                "link": f"https://www.youtube.com/watch?v={e.get('id')}",
                "duration": e.get("duration"),
            }
            for e in data.get("entries", [])
        ]

    # ----------------------------------------------------
    # 2) FETCH SEGMENTS — CAPTIONS FIRST, WHISPER FALLBACK
    # ----------------------------------------------------
    def _fetch_segments(self, video_id: str):
        """returns [{'start':..., 'end':..., 'text':...}]"""

        # Try official YouTube transcripts
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            t = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                tr = t.find_transcript(["en", "en-US", "en-GB"])
            except:
                tr = next(iter(t._transcripts.values()))

            fetched = tr.fetch()
            return [
                {
                    "start": float(s["start"]),
                    "end": float(s["start"] + s["duration"]),
                    "text": s["text"].strip(),
                }
                for s in fetched if s.get("text")
            ]
        except:
            pass  # fallback → whisper

        # Whisper fallback
        from yt_dlp import YoutubeDL
        whisper_model = self._get_whisper()

        audio_dir = os.path.join(os.getcwd(), "audio_temp")
        os.makedirs(audio_dir, exist_ok=True)

        outtmpl = os.path.join(audio_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
            ],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        audio_path = os.path.join(audio_dir, f"{video_id}.mp3")
        result = whisper_model.transcribe(audio_path)

        return [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in result.get("segments", [])
        ]

    # ----------------------------------------------------
    # 3) CHUNK
    # ----------------------------------------------------
    def _chunk_segments(self, segments, max_words=300):
        chunks = []
        buf = []
        current_start = None

        for s in segments:
            words = s["text"].split()
            if not buf:
                current_start = s["start"]

            if len(buf) + len(words) > max_words:
                chunks.append((" ".join(buf), current_start))
                buf = []
                current_start = s["start"]

            buf.extend(words)

        if buf:
            chunks.append((" ".join(buf), current_start))

        return chunks

    # ----------------------------------------------------
    # 4) PREPARE VIDEO (CACHE)
    # ----------------------------------------------------
    def prepare_video(self, video_id: str):
        if video_id in self._videos:
            return

        segments = self._fetch_segments(video_id)
        if not segments:
            raise RuntimeError("No transcript available")

        chunks = self._chunk_segments(segments)
        texts = [c[0] for c in chunks]
        starts = [c[1] for c in chunks]

        embedder = self._get_embedder()
        vectors = embedder.encode(texts, convert_to_numpy=True).astype("float32")

        import faiss
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        self._videos[video_id] = {
            "chunks": texts,
            "starts": starts,
            "index": index,
            "vectors": vectors,
        }

    # ----------------------------------------------------
    # 5) RETRIEVE
    # ----------------------------------------------------
    def _retrieve(self, video_id, question, top_k=5, timestamp=None):
        self.prepare_video(video_id)
        rec = self._videos[video_id]

        embedder = self._get_embedder()
        q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")

        distances, idx = rec["index"].search(q_vec, top_k)
        chunks = rec["chunks"]

        results = [chunks[int(i)] for i in idx[0]]

        # Timestamp bias
        if timestamp is not None:
            nearest = min(range(len(rec["starts"])), key=lambda i: abs(rec["starts"][i] - timestamp))
            if chunks[nearest] not in results:
                results.insert(0, chunks[nearest])
                results = results[:top_k]

        return results

    # ----------------------------------------------------
    # 6) PUBLIC: ASK VIDEO
    # ----------------------------------------------------
    def ask_video(self, question, video_id, timestamp=None, top_k=5):
        context = self._retrieve(video_id, question, top_k, timestamp)
        context_text = "\n\n".join(context)

        prompt = f"""
You are a friendly science tutor for class 6–10 students.
Use the video transcript context to answer clearly and simply.

If the answer is not in the transcript, say so briefly.

Context:
{context_text}

Student question: {question}

Answer:
"""
        return self._llm_call(prompt).strip()
