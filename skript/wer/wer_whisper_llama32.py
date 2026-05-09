from wer_base import run

run(
    stt_filter='whisper',
    llm_filter='llama3.2',
    title='Whisper large-v3-turbo + llama3.2',
    out_file='wer_whisper_llama32.md',
)
