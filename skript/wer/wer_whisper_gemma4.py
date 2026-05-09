from wer_base import run

run(
    stt_filter='whisper',
    llm_filter='gemma4',
    title='Whisper large-v3-turbo + gemma4',
    out_file='wer_whisper_gemma4.md',
)
