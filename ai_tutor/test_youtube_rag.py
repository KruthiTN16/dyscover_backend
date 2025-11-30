# test_youtube_rag.py
"""
CLI to test the new backend.
Flow:
 - Enter search topic
 - Choose a video
 - Prepare it (build index) -> done once
 - Ask questions; optional timestamp after '|' to bias e.g. "What is chlorophyll? | 42.5"
"""

from ai_tutor.rag_youtube import YouTubeRAG


def main():
    rag = YouTubeRAG()  # uses default small llm (change in constructor if needed)
    print("YouTubeRAG CLI tester. Type 'exit' to quit.\n")

    while True:
        topic = input("Enter topic to search (or 'exit'): ").strip()
        if topic.lower() == 'exit':
            break
        if not topic:
            continue

        results = rag.search_youtube(topic, max_results=5)
        if not results:
            print("No videos found. Try another topic.")
            continue

        print("\nTop results:")
        for i, v in enumerate(results[:5], start=1):
            print(f"{i}. {v['title']} [{v['channel']}] ({v['duration']}s) id:{v['video_id']}")

        sel = input("Select index (1) or press Enter for top result: ").strip()
        try:
            idx = int(sel) - 1 if sel else 0
            chosen = results[idx]
        except Exception:
            chosen = results[0]

        print(f"\nSelected: {chosen['title']} ({chosen['video_id']})")
        ok = input("Prepare this video for Q&A? (y/n): ").strip().lower()
        if ok != 'y':
            continue

        print("Preparing video (fetching transcript, chunking, embeddings) — may take a moment...")
        try:
            rag.prepare_video(chosen['video_id'])
            print("Prepared. You can now ask questions.")
        except Exception as e:
            print("Failed to prepare video:", e)
            continue

        # Chat loop
        while True:
            q = input("\nAsk question (or 'back'/'exit'). Optionally add '| <seconds>' to bias to time.\n> ").strip()
            if not q:
                continue
            if q.lower() in ('back', 'exit'):
                break

            timestamp = None
            if '|' in q:
                parts = q.split('|', 1)
                q_text = parts[0].strip()
                try:
                    timestamp = float(parts[1].strip())
                except Exception:
                    timestamp = None
            else:
                q_text = q

            try:
                ans = rag.ask_video(q_text, chosen['video_id'], timestamp=timestamp, top_k=5)
                print("\n--- AI Answer ---\n")
                print(ans)
                print("\n-----------------\n")
            except Exception as e:
                print("Error answering:", e)
                break

        if q.lower() == 'exit':
            break

if __name__ == "__main__":
    main()
