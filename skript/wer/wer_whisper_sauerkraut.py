from wer_base import run

run(
    stt_filter='whisper',
    llm_filter='sauerkraut',
    title='Whisper large-v3-turbo + SauerkrautLM 8b',
    out_file='wer_whisper_sauerkraut.md',
)
