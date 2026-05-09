from llm_check_base import run

run(
    stt_filter='whisper',
    llm_filter='gemma4',
    title='Whisper large-v3-turbo + gemma4',
    out_file='llm_check_whisper_gemma4.md',
)
